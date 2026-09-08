// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

// rosterGetPair stands up two cross-pinned managers, the only state the roster
// endpoint is ever reached in.
func rosterGetPair(t *testing.T, portA, portB int) (*Manager, *Manager, string) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	server, client := newTestManagerPort(t, portA), newTestManagerPort(t, portB)
	go func() { _ = server.runHTTP(ctx) }()
	go func() { _ = client.runHTTP(ctx) }()
	time.Sleep(400 * time.Millisecond)
	pinTrusted(t, server, client.identity.NodeUUID, string(client.identity.CertPEM), client.identity.CertFingerprint)
	pinTrusted(t, client, server.identity.NodeUUID, string(server.identity.CertPEM), server.identity.CertFingerprint)
	server.upsertMember(&ClusterNode{NodeUUID: client.identity.NodeUUID, ID: "cli", IPAddress: "127.0.0.1", Port: portB, AdmissionEpoch: 1, State: stateMember})
	client.upsertMember(&ClusterNode{NodeUUID: server.identity.NodeUUID, ID: "srv", IPAddress: "127.0.0.1", Port: portA, AdmissionEpoch: 1, State: stateMember})
	return server, client, net.JoinHostPort("127.0.0.1", strconv.Itoa(portA))
}

func publishDescriptor(t *testing.T, m *Manager, ll int) {
	t.Helper()
	raw, err := json.Marshal(noderec.DirectConnectParams{
		HostUUID: m.identity.NodeUUID,
		Services: noderec.ServiceMap{
			noderec.ServiceLlamaCpp: ll, noderec.ServiceEngineManager: 14322,
			noderec.ServiceEngineControl: 14323, noderec.ServiceCluster: 14321,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	m.applyDirectConnect(&Message{Params: raw})
	if m.directConnectDescriptor() == nil {
		t.Fatal("precondition: no descriptor published")
	}
}

func do(t *testing.T, c *http.Client, method, addr string, body []byte) (*http.Response, []byte) {
	t.Helper()
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequest(method, "https://"+addr+rosterPath, rdr)
	if err != nil {
		t.Fatal(err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.Do(req)
	if err != nil {
		t.Fatalf("%s: %v", method, err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	return resp, b
}

// TestRosterGetServesDescriptorToPinnedPeer is the regression for the deployed
// defect: manual-nodes bootstraps with GET, the handler answered 405, so every
// probe classified authentication_failed and suppressed the service map.
func TestRosterGetServesDescriptorToPinnedPeer(t *testing.T) {
	for _, ll := range []int{8081, 18435} {
		server, client, addr := rosterGetPair(t, 15300+ll%100, 15400+ll%100)
		publishDescriptor(t, server, ll)
		hc, err := client.peerClient(server.identity.NodeUUID)
		if err != nil {
			t.Fatal(err)
		}
		resp, body := do(t, hc, http.MethodGet, addr, nil)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("ll=%d status %d, want 200", ll, resp.StatusCode)
		}
		var r Roster
		if err := json.Unmarshal(body, &r); err != nil {
			t.Fatalf("decode: %v", err)
		}
		dc := r.DirectConnect
		if dc == nil {
			t.Fatalf("ll=%d authenticated GET returned no descriptor", ll)
		}
		if dc.HostUUID != server.identity.NodeUUID || dc.Principal != server.identity.NodeUUID {
			t.Fatalf("identity binding broken: %+v", dc)
		}
		if _, ok := client.trust.Get(dc.Principal); !ok {
			t.Fatal("principal is not the pinned NodeUUID")
		}
		if dc.Services[noderec.ServiceLlamaCpp] != ll {
			t.Fatalf("ll port %d, want %d verbatim", dc.Services[noderec.ServiceLlamaCpp], ll)
		}
		if dc.Services[noderec.ServiceEngineManager] != 14322 || dc.Services[noderec.ServiceEngineControl] != 14323 {
			t.Fatalf("em/ec missing: %v", dc.Services)
		}
	}
}

func TestRosterGetRefusesUnauthenticatedAndUnpinned(t *testing.T) {
	server, _, addr := rosterGetPair(t, 15205, 15206)
	publishDescriptor(t, server, 8081)

	bare := &http.Client{Timeout: 5 * time.Second,
		Transport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}}
	if resp, err := bare.Get("https://" + addr + rosterPath); err == nil {
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			t.Fatal("anonymous GET was served the roster")
		}
	}

	stranger := newTestManagerPort(t, 15207)
	pinTrusted(t, stranger, server.identity.NodeUUID, string(server.identity.CertPEM), server.identity.CertFingerprint)
	hc, err := stranger.peerClient(server.identity.NodeUUID)
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodGet, "https://"+addr+rosterPath, nil)
	if resp, err := hc.Do(req); err == nil {
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Fatalf("unpinned GET status %d, want 403", resp.StatusCode)
		}
	}
	if _, ok := server.trust.Get(stranger.identity.NodeUUID); ok {
		t.Fatal("an unpinned GET caused the server to pin the caller")
	}
}

func TestRosterGetIsNonMutatingAndPostStillWorks(t *testing.T) {
	server, client, addr := rosterGetPair(t, 15208, 15209)
	publishDescriptor(t, server, 8081)
	snap := func() string {
		b, err := json.Marshal(map[string]any{
			"nodes": server.snapshotNodes(), "pins": len(server.trust.List()),
			"id": server.identity.NodeUUID, "desc": server.directConnectDescriptor(),
			"tombs": server.snapshotTombstones(), "proofs": server.snapshotRemovalProofs(),
		})
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	hc, err := client.peerClient(server.identity.NodeUUID)
	if err != nil {
		t.Fatal(err)
	}
	before := snap()
	for i := 0; i < 3; i++ {
		if resp, _ := do(t, hc, http.MethodGet, addr, nil); resp.StatusCode != http.StatusOK {
			t.Fatalf("read %d: %d", i, resp.StatusCode)
		}
	}
	if snap() != before {
		t.Fatal("authenticated GET mutated server state")
	}

	body, err := json.Marshal(client.buildLocalRoster())
	if err != nil {
		t.Fatal(err)
	}
	if resp, _ := do(t, hc, http.MethodPost, addr, body); resp.StatusCode != http.StatusOK {
		t.Fatalf("POST reconcile status %d, want 200", resp.StatusCode)
	}
	resp, _ := do(t, hc, http.MethodPut, addr, nil)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("PUT status %d, want 405", resp.StatusCode)
	}
	if got := resp.Header.Get("Allow"); got != "GET, POST" {
		t.Fatalf("Allow %q, want \"GET, POST\"", got)
	}
}
