// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"sync"

	"nvpair-shared/noderec"
)

const (
	clusterManagerPort = 14321
	rosterPath         = "/v1/cluster/roster"
	maxRosterBytes     = 1 << 20
	nodeURISANPrefix   = "urn:nvpair:node:"
)

// directConnectOutcome is the tri-state result of the bootstrap. The three
// states are not interchangeable: only an ABSENT descriptor may preserve the
// legacy partial behavior of an old peer, while a FAILED authentication must
// suppress every legacy claim, because a peer that answers but cannot be
// authenticated is precisely the case legacy probing would misreport.
type directConnectOutcome int

const (
	directConnectAbsent directConnectOutcome = iota
	directConnectAccepted
	directConnectFailed
)

func (o directConnectOutcome) String() string {
	switch o {
	case directConnectAccepted:
		return "accepted"
	case directConnectFailed:
		return "authentication_failed"
	default:
		return "descriptor_absent"
	}
}

type rosterResponse struct {
	ClusterID     string                 `json:"clusterId"`
	DirectConnect *noderec.DirectConnect `json:"directConnect,omitempty"`
}

// authenticateDirectConnect performs the same-channel direct-connect bootstrap.
//
// Plaintext node-info contributes exactly two hints and no authority: which
// principal to select a local pin for, and optionally which port the peer's
// cluster-manager listens on. The route host is always the configured seed, and
// every service used for inventory or routing comes from the authenticated
// descriptor only.
func (m *Manager) authenticateDirectConnect(addr string, info NodeInfoResponse, priorCluster string) (noderec.ServiceMap, directConnectOutcome) {
	if info.ClusterUUID == nil || *info.ClusterUUID == "" {
		if priorCluster != "" {
			// Case 4: this node has asserted a cluster identity before, so an
			// answer that now omits it is a downgrade attempt, not a standalone
			// host. Fail closed rather than reverting to legacy behavior.
			slog.Warn("manual node dropped its cluster annotation; treating as downgrade",
				"addr", addr, "previously", priorCluster)
			return nil, directConnectFailed
		}
		// Case 1: genuinely standalone manual node. Legacy behavior is preserved
		// by the caller; this path creates no authenticated claim and no valid
		// service map.
		return nil, directConnectAbsent
	}
	principal := *info.ClusterUUID
	if info.HostUUID == "" || info.HostUUID != principal {
		return nil, directConnectFailed
	}
	m.mesh.Refresh()
	if !m.mesh.Clustered() {
		// FAIL CLOSED: with no local pins the selected principal cannot be
		// authenticated at all, so legacy fallback must not be reachable.
		return nil, directConnectFailed
	}
	cfg, ok := m.mesh.ClientTLSConfig(principal)
	if !ok || cfg == nil {
		return nil, directConnectFailed
	}
	port := clusterManagerPort
	if hinted, ok := info.Services[noderec.ServiceCluster]; ok && hinted >= 1 && hinted <= 65535 {
		port = hinted
	}
	target := net.JoinHostPort(addr, strconv.Itoa(port))
	conn, err := tls.DialWithDialer(&net.Dialer{Timeout: probeTimeout}, "tcp", target, cfg)
	if err != nil {
		slog.Debug("direct-connect bootstrap TLS failed", "addr", target, "principal", principal, "err", err)
		return nil, directConnectFailed
	}
	defer conn.Close()
	if !leafNamesPrincipal(conn, principal) {
		slog.Warn("direct-connect leaf does not name the selected principal", "addr", target, "principal", principal)
		return nil, directConnectFailed
	}
	desc, err := fetchDirectConnect(conn, addr, port)
	if err != nil {
		slog.Debug("direct-connect roster fetch failed", "addr", target, "err", err)
		return nil, directConnectFailed
	}
	if desc == nil {
		return nil, directConnectAbsent
	}
	if !desc.BoundTo(principal) {
		slog.Warn("direct-connect descriptor is not bound to the authenticated principal", "addr", target, "principal", principal)
		return nil, directConnectFailed
	}
	return desc.Services.Clone(), directConnectAccepted
}

// leafNamesPrincipal confirms the authenticated leaf carries the selected
// principal, so another pinned member of the same cluster cannot stand in.
func leafNamesPrincipal(conn *tls.Conn, principal string) bool {
	chain := conn.ConnectionState().PeerCertificates
	if len(chain) == 0 {
		return false
	}
	want := nodeURISANPrefix + principal
	for _, uri := range chain[0].URIs {
		if uri != nil && uri.String() == want {
			return true
		}
	}
	return chain[0].Subject.CommonName == principal
}

func fetchDirectConnect(conn *tls.Conn, host string, port int) (*noderec.DirectConnect, error) {
	url := "https://" + net.JoinHostPort(host, strconv.Itoa(port)) + rosterPath
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if err := req.Write(conn); err != nil {
		return nil, err
	}
	resp, err := http.ReadResponse(bufio.NewReader(conn), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("roster status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxRosterBytes))
	if err != nil {
		return nil, err
	}
	var roster rosterResponse
	if err := json.Unmarshal(body, &roster); err != nil {
		return nil, err
	}
	return roster.DirectConnect, nil
}

// clusterMemory remembers the last nonempty cluster identity each manual node
// asserted over plaintext. It exists to separate a genuinely standalone host
// (never annotated) from a node that HAD an identity and then stopped
// presenting one, which is a downgrade attempt. Memory-only and per-process:
// like the descriptor itself it is a live observation, and a persisted copy
// would be a claim rather than something this process saw.
type clusterMemory struct {
	mu   sync.Mutex
	seen map[string]string
}

var directConnectMemory = &clusterMemory{seen: make(map[string]string)}

func (c *clusterMemory) prior(id string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.seen[id]
}

func (c *clusterMemory) remember(id, cluster string) {
	if id == "" || cluster == "" {
		return
	}
	c.mu.Lock()
	c.seen[id] = cluster
	c.mu.Unlock()
}

func (c *clusterMemory) forget(id string) {
	c.mu.Lock()
	delete(c.seen, id)
	c.mu.Unlock()
}
