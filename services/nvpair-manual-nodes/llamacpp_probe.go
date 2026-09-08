// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"crypto/tls"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
)

// llamaCppProbeTimeout bounds the authenticated inventory fetch. It is longer
// than probeTimeout because this is a model-list read on a peer that may be
// mid-generation, not a liveness poke; the probe loop is 10s, so this still
// cannot overlap its own next tick.
const llamaCppProbeTimeout = 5 * probeTimeout

// maxLlamaCppModelsBytes caps the inventory body. A peer's model list is small
// and bounded; anything larger is a malformed or hostile response, and the cap
// keeps a prober from being made to buffer arbitrary bytes by a machine whose
// only qualification is holding a pin we accept.
const maxLlamaCppModelsBytes = 1 << 20

// llamaCppModelsPath is engine-manager's OpenAI-shaped inventory route, served
// on the llama.cpp proxy's cluster-scoped mTLS ingress.
const llamaCppModelsPath = "/v1/models"

// llamaCppEngine is the engine-manager engine name llama.cpp models are
// attributed to. It matches the key discovered nodes use, so both discovery
// sources present ModelsByEngine identically.
const llamaCppEngine = "llamacpp"

// principalFor returns the cluster principal a node-info response asserts, or
// "" when it asserts none or is too old to report the field. A pointer that is
// present-and-empty is a node stating it belongs to no cluster; both read as no
// principal to select a pin for, which is the only thing this value is used for.
func principalFor(info NodeInfoResponse) string {
	if info.ClusterUUID == nil {
		return ""
	}
	return *info.ClusterUUID
}

// probeLlamaCppAuthenticated reads a peer's llama.cpp inventory over the pin
// already selected for principal, on the port the AUTHENTICATED descriptor
// named. It is called only on the accepted branch, so an absent or failed
// descriptor never reaches it.
//
// This is the same-channel rule the direct-connect bootstrap established, for
// the same reason: plaintext em refuses non-loopback callers, so the only
// inventory a consumer may trust is one read over a connection whose far end
// proved it holds the principal's key. The leaf is re-checked here rather than
// assumed from the earlier roster fetch, because that was a separate connection
// and nothing carries its authentication forward.
//
// Returns up=false with no models on any failure. A peer that answers but
// cannot be authenticated contributes nothing, exactly as elsewhere.
func (m *Manager) probeLlamaCppAuthenticated(addr, principal string, port int) (bool, []string, []string) {
	if addr == "" || principal == "" || port < 1 || port > 65535 {
		return false, nil, nil
	}

	m.mesh.Refresh()
	if !m.mesh.Clustered() {
		// No local cluster state means the principal cannot be authenticated at
		// all, so there is no fallback to reach for.
		return false, nil, nil
	}
	cfg, ok := m.mesh.ClientTLSConfig(principal)
	if !ok || cfg == nil {
		slog.Debug("llama.cpp inventory: no pin for the selected principal",
			"addr", addr, "principal", principal)
		return false, nil, nil
	}

	target := net.JoinHostPort(addr, strconv.Itoa(port))
	conn, err := tls.DialWithDialer(&net.Dialer{Timeout: llamaCppProbeTimeout}, "tcp", target, cfg)
	if err != nil {
		slog.Debug("llama.cpp inventory TLS failed", "target", target, "err", err)
		return false, nil, nil
	}
	defer conn.Close()

	if !leafNamesPrincipal(conn, principal) {
		slog.Warn("llama.cpp inventory leaf does not name the selected principal",
			"target", target, "principal", principal)
		return false, nil, nil
	}

	req, err := http.NewRequest(http.MethodGet, "https://"+target+llamaCppModelsPath, nil)
	if err != nil {
		return false, nil, nil
	}
	req.Header.Set("Accept", "application/json")
	if err := req.Write(conn); err != nil {
		slog.Debug("llama.cpp inventory write failed", "target", target, "err", err)
		return false, nil, nil
	}
	resp, err := http.ReadResponse(bufio.NewReader(conn), req)
	if err != nil {
		slog.Debug("llama.cpp inventory read failed", "target", target, "err", err)
		return false, nil, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		slog.Debug("llama.cpp inventory status", "target", target, "status", resp.StatusCode)
		return false, nil, nil
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxLlamaCppModelsBytes))
	if err != nil {
		return false, nil, nil
	}

	models, loaded := parseLlamaCppModels(body)
	return true, models, loaded
}

// llamaCppModelsEnvelope is the subset of the inventory body this needs. Every
// other field the peer sends is ignored rather than mirrored: a model list is
// the only thing being asked for, and carrying process arguments or file paths
// onward would put a peer's local detail into this node's directory.
type llamaCppModelsEnvelope struct {
	Data []struct {
		ID      string `json:"id"`
		OwnedBy string `json:"owned_by"`
		Status  *struct {
			Value string `json:"value"`
		} `json:"status"`
	} `json:"data"`
}

// parseLlamaCppModels extracts the llama.cpp model ids and the subset currently
// resident in memory. An entry with no owned_by is accepted as llama.cpp because
// that is the engine whose port was dialed; an entry naming a DIFFERENT engine
// is skipped, so a mixed inventory cannot attribute another engine's models to
// llama.cpp. Order is preserved and duplicates are dropped.
//
// Kept a pure function of the body so it is testable without a TLS peer.
func parseLlamaCppModels(body []byte) ([]string, []string) {
	var env llamaCppModelsEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, nil
	}
	var models, loaded []string
	seen := make(map[string]bool, len(env.Data))
	for _, entry := range env.Data {
		if entry.ID == "" || seen[entry.ID] {
			continue
		}
		if entry.OwnedBy != "" && entry.OwnedBy != llamaCppEngine {
			continue
		}
		seen[entry.ID] = true
		models = append(models, entry.ID)
		if entry.Status != nil && entry.Status.Value == "loaded" {
			loaded = append(loaded, entry.ID)
		}
	}
	return models, loaded
}

// stringSliceEqual reports order-sensitive equality, with nil and empty equal.
// Order-sensitive is deliberate: the inventory arrives in the peer's own order
// and a reordering is a real change worth re-emitting.
func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
