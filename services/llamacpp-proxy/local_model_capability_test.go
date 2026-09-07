// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

const (
	exactCapability   = "DeepSeek-V4-Pro-Qwen3.5-4B-MTP-Q4_K_M"
	compactCapability = "DeepSeek-V4-Pro-Qwen3.5-4B-MTP-Q4KM"
)

func mustCapabilityURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	value, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestCapabilityOverlayRetainsSubscribedMetadata(t *testing.T) {
	proxy := &Proxy{discovery: NewDiscovery()}
	subscribed := Node{
		ID:          "self",
		Host:        "macbook-pro.local",
		Port:        18435,
		Addresses:   []string{"192.168.1.71"},
		TXT:         []string{"lm=18435"},
		Models:      []string{compactCapability},
		ClusterUUID: "cluster-principal",
	}
	manual := Node{
		ID:        "self",
		Host:      "127.0.0.1",
		Port:      18434,
		Addresses: []string{"127.0.0.1"},
		Models:    []string{exactCapability},
	}

	proxy.discovery.SetSubscribed([]Node{subscribed})
	proxy.discovery.AddManual(manual)

	got := proxy.overlayAuthoritativeLocalModels(
		proxy.discovery.Nodes(),
		mustCapabilityURL(t, "http://127.0.0.1:18434"),
		true,
	)

	if len(got) != 1 {
		t.Fatalf("got %d nodes, want 1", len(got))
	}
	if got[0].Host != subscribed.Host ||
		got[0].Port != subscribed.Port ||
		got[0].ClusterUUID != subscribed.ClusterUUID ||
		!reflect.DeepEqual(got[0].Addresses, subscribed.Addresses) ||
		!reflect.DeepEqual(got[0].TXT, subscribed.TXT) {
		t.Fatalf("subscribed metadata changed: %#v", got[0])
	}
	if !reflect.DeepEqual(got[0].Models, manual.Models) {
		t.Fatalf("models = %#v, want %#v", got[0].Models, manual.Models)
	}
	if nodeAdvertisesModel(got[0], compactCapability) {
		t.Fatal("Q4KM unexpectedly matched Q4_K_M")
	}
	if !nodeAdvertisesModel(got[0], exactCapability) {
		t.Fatal("exact Q4_K_M did not match")
	}
}

func TestCapabilityOverlayEmptyClearsStale(t *testing.T) {
	proxy := &Proxy{discovery: NewDiscovery()}
	proxy.discovery.SetSubscribed([]Node{{
		ID:     "self",
		Models: []string{"stale-model"},
	}})
	proxy.discovery.AddManual(Node{
		ID:     "self",
		Host:   "127.0.0.1",
		Port:   18434,
		Models: []string{},
	})

	got := proxy.overlayAuthoritativeLocalModels(
		proxy.discovery.Nodes(),
		mustCapabilityURL(t, "http://127.0.0.1:18434"),
		true,
	)
	if len(got) != 1 || got[0].Models == nil || len(got[0].Models) != 0 {
		t.Fatalf("empty inventory did not clear stale models: %#v", got)
	}
}

func TestCapabilityOverlayRejectsNonLocalManualRecords(t *testing.T) {
	tests := []struct {
		name      string
		manual    Node
		target    *url.URL
		hasTarget bool
	}{
		{
			name: "wrong host",
			manual: Node{
				ID: "node", Host: "192.168.1.72", Port: 18434,
				Models: []string{"manual-model"},
			},
			target:    mustCapabilityURL(t, "http://127.0.0.1:18434"),
			hasTarget: true,
		},
		{
			name: "wrong port",
			manual: Node{
				ID: "node", Host: "127.0.0.1", Port: 18436,
				Models: []string{"manual-model"},
			},
			target:    mustCapabilityURL(t, "http://127.0.0.1:18434"),
			hasTarget: true,
		},
		{
			name: "no local backend",
			manual: Node{
				ID: "node", Host: "127.0.0.1", Port: 18434,
				Models: []string{"manual-model"},
			},
			hasTarget: false,
		},
		{
			name: "same ID remote manual",
			manual: Node{
				ID: "node", Host: "192.168.1.72", Port: 18435,
				Models: []string{"manual-model"},
			},
			target:    mustCapabilityURL(t, "http://127.0.0.1:18434"),
			hasTarget: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proxy := &Proxy{discovery: NewDiscovery()}
			proxy.discovery.SetSubscribed([]Node{{
				ID:          "node",
				Host:        "peer.local",
				Port:        18435,
				Models:      []string{"subscribed-model"},
				ClusterUUID: "peer-principal",
			}})
			proxy.discovery.AddManual(tt.manual)

			got := proxy.overlayAuthoritativeLocalModels(
				proxy.discovery.Nodes(),
				tt.target,
				tt.hasTarget,
			)
			if !reflect.DeepEqual(got[0].Models, []string{"subscribed-model"}) {
				t.Fatalf("non-local manual record overrode subscribed models: %#v", got[0])
			}
		})
	}
}

func TestDiscoveryManualDefensiveCopies(t *testing.T) {
	discovery := NewDiscovery()
	discovery.AddManual(Node{
		ID:        "node",
		Addresses: []string{"127.0.0.1"},
		TXT:       []string{"lm=18435"},
		Models:    []string{exactCapability},
	})

	first, ok := discovery.Manual("node")
	if !ok {
		t.Fatal("manual node missing")
	}
	first.Addresses[0] = "mutated"
	first.TXT[0] = "mutated"
	first.Models[0] = "mutated"

	second, ok := discovery.Manual("node")
	if !ok {
		t.Fatal("manual node missing after mutation")
	}
	if second.Addresses[0] != "127.0.0.1" ||
		second.TXT[0] != "lm=18435" ||
		second.Models[0] != exactCapability {
		t.Fatalf("stored manual node mutated: %#v", second)
	}
}

func TestCapabilityOverlayAuthenticatedHTTPRouting(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "local-destination-token")
	writeLocalAuthToken(t, tokenPath, "destination-token")
	writeLocalAuthConfig(t, configPath, tokenPath)

	var exactHits int
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		exactHits++

		if got := r.Header.Get("Authorization"); got != "Bearer destination-token" {
			t.Errorf("local backend Authorization = %q, want destination bearer", got)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if strings.Contains(r.Header.Get("Authorization"), "caller-token") {
			t.Error("caller Authorization reached local backend")
		}
		if got := r.Header.Get("Cookie"); got != "" {
			t.Errorf("caller Cookie reached local backend: %q", got)
		}
		if got := r.Header.Get("Connection"); got != "" {
			t.Errorf("hop-by-hop Connection header reached local backend: %q", got)
		}
		if got := r.Header.Get("Keep-Alive"); got != "" {
			t.Errorf("hop-by-hop Keep-Alive header reached local backend: %q", got)
		}
		if got := r.Header.Get("Proxy-Connection"); got != "" {
			t.Errorf("hop-by-hop Proxy-Connection header reached local backend: %q", got)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"exact-local"}}]}`))
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}
	backendPort, err := strconv.Atoi(backendURL.Port())
	if err != nil {
		t.Fatal(err)
	}

	discovery := NewDiscovery()
	discovery.SetSubscribed([]Node{{
		ID:          "self",
		Host:        "127.0.0.1",
		Port:        18435,
		Addresses:   []string{"127.0.0.1"},
		ClusterUUID: "self-principal",
		Models:      []string{compactCapability},
	}})
	discovery.AddManual(Node{
		ID:        "self",
		Host:      backendURL.Hostname(),
		Port:      backendPort,
		Addresses: []string{backendURL.Hostname()},
		Models:    []string{exactCapability},
	})

	proxy := testProxy(discovery, 18435)
	proxy.setLocalBackend(localBackend{
		Engine:  "llamacpp",
		Host:    backendURL.Hostname(),
		Port:    backendPort,
		Healthy: true,
	})
	proxy.SetSelected("self")

	exactRequest := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"`+exactCapability+`"}`),
	)
	exactRequest.Header.Set("Authorization", "Bearer caller-token")
	exactRequest.Header.Set("Cookie", "session=caller-cookie")
	exactRequest.Header.Set("Connection", "Authorization, Keep-Alive, Proxy-Connection")
	exactRequest.Header.Set("Keep-Alive", "timeout=5")
	exactRequest.Header.Set("Proxy-Connection", "keep-alive")
	exactResponse := httptest.NewRecorder()

	proxy.handleHTTP(exactResponse, exactRequest)

	if exactResponse.Code != http.StatusOK {
		t.Fatalf(
			"exact-model status = %d, want 200; body=%q",
			exactResponse.Code,
			exactResponse.Body.String(),
		)
	}
	if exactHits != 1 {
		t.Fatalf("exact local backend hits = %d, want 1", exactHits)
	}
	if !strings.Contains(exactResponse.Body.String(), "exact-local") {
		t.Fatalf(
			"exact response body = %q, want direct local backend response",
			exactResponse.Body.String(),
		)
	}

	// The subscribed Q4KM record is stale. It must not fuzzy-match the
	// authoritative local Q4_K_M model, and the rejected request must not
	// reach the local backend or self-forward through the proxy listener.
	exactHits = 0
	compactRequest := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"`+compactCapability+`"}`),
	)
	compactRequest.Header.Set("Authorization", "Bearer caller-token")
	compactResponse := httptest.NewRecorder()

	proxy.handleHTTP(compactResponse, compactRequest)

	if compactResponse.Code != http.StatusBadGateway {
		t.Fatalf(
			"Q4KM status = %d, want 502 local capability rejection; body=%q",
			compactResponse.Code,
			compactResponse.Body.String(),
		)
	}
	if exactHits != 0 {
		t.Fatalf(
			"Q4KM request reached local backend %d times; want zero",
			exactHits,
		)
	}
	if !strings.Contains(
		compactResponse.Body.String(),
		"no available node advertises the requested model",
	) {
		t.Fatalf(
			"Q4KM rejection body = %q, want actionable local capability rejection",
			compactResponse.Body.String(),
		)
	}
}

func TestEndpointAwareManualClassification(t *testing.T) {
	tests := []struct {
		name     string
		node     Node
		manual   Node
		selected string
		want     bool
	}{
		{
			name: "manual-only exact endpoint",
			node: Node{
				ID: "manual-only", Host: "127.0.0.1", Port: 19101,
				Addresses: []string{"127.0.0.1"},
			},
			manual: Node{
				ID: "manual-only", Host: "127.0.0.1", Port: 19101,
				Addresses: []string{"127.0.0.1"},
			},
			selected: "http://127.0.0.1:19101",
			want:     true,
		},
		{
			name: "same ID subscribed different endpoint",
			node: Node{
				ID: "shared", Host: "192.0.2.71", Port: 18435,
				Addresses: []string{"192.0.2.71"},
			},
			manual: Node{
				ID: "shared", Host: "127.0.0.1", Port: 19101,
				Addresses: []string{"127.0.0.1"},
			},
			selected: "http://192.0.2.71:18435",
			want:     false,
		},
		{
			name: "wrong host",
			node: Node{
				ID: "wrong-host", Host: "192.0.2.72", Port: 19102,
				Addresses: []string{"192.0.2.72"},
			},
			manual: Node{
				ID: "wrong-host", Host: "127.0.0.1", Port: 19102,
				Addresses: []string{"127.0.0.1"},
			},
			selected: "http://192.0.2.72:19102",
			want:     false,
		},
		{
			name: "wrong port",
			node: Node{
				ID: "wrong-port", Host: "127.0.0.1", Port: 19104,
				Addresses: []string{"127.0.0.1"},
			},
			manual: Node{
				ID: "wrong-port", Host: "127.0.0.1", Port: 19103,
				Addresses: []string{"127.0.0.1"},
			},
			selected: "http://127.0.0.1:19104",
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			discovery := NewDiscovery()
			discovery.AddManual(tt.manual)
			proxy := testProxy(discovery, 18435)

			selected := mustCapabilityURL(t, tt.selected)
			if got := proxy.isManualCandidateEndpoint(tt.node, selected); got != tt.want {
				t.Fatalf("manual classification = %t, want %t", got, tt.want)
			}
		})
	}
}

func TestEndpointAwareManualOnlyCandidateStillRoutes(t *testing.T) {
	var hits int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	serverURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(serverURL.Port())
	if err != nil {
		t.Fatal(err)
	}

	discovery := NewDiscovery()
	discovery.AddManual(Node{
		ID:        "manual-only",
		Host:      serverURL.Hostname(),
		Port:      port,
		Addresses: []string{serverURL.Hostname()},
		Models:    []string{exactCapability},
	})

	proxy := testProxy(discovery, 18435)
	proxy.SetSelected("manual-only")

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"`+exactCapability+`"}`),
	)
	response := httptest.NewRecorder()
	proxy.handleHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%q", response.Code, response.Body.String())
	}
	if hits != 1 {
		t.Fatalf("manual backend hits = %d, want 1", hits)
	}
}

func TestEndpointAwareSameIDUnpinnedSubscribedIsNotDialedAsManual(t *testing.T) {
	var manualHits int
	manualServer := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		manualHits++
	}))
	defer manualServer.Close()

	manualURL, err := url.Parse(manualServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	manualPort, err := strconv.Atoi(manualURL.Port())
	if err != nil {
		t.Fatal(err)
	}

	discovery := NewDiscovery()
	discovery.SetSubscribed([]Node{{
		ID:        "shared-unpinned",
		Host:      "192.0.2.72",
		Port:      18435,
		Addresses: []string{"192.0.2.72"},
		Models:    []string{exactCapability},
	}})
	discovery.AddManual(Node{
		ID:        "shared-unpinned",
		Host:      manualURL.Hostname(),
		Port:      manualPort,
		Addresses: []string{manualURL.Hostname()},
		Models:    []string{exactCapability},
	})

	proxy := testProxy(discovery, 18435)
	proxy.SetSelected("shared-unpinned")

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"`+exactCapability+`"}`),
	)
	response := httptest.NewRecorder()
	proxy.handleHTTP(response, request)

	if manualHits != 0 {
		t.Fatalf("manual endpoint hits = %d, want zero", manualHits)
	}
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body=%q", response.Code, response.Body.String())
	}
}

func TestEndpointAwareSameIDPinnedSubscribedIsNotManual(t *testing.T) {
	discovery := NewDiscovery()
	discovery.AddManual(Node{
		ID:        "shared-pinned",
		Host:      "127.0.0.1",
		Port:      19103,
		Addresses: []string{"127.0.0.1"},
		Models:    []string{exactCapability},
	})

	proxy := testProxy(discovery, 18435)
	subscribed := Node{
		ID:          "shared-pinned",
		Host:        "192.0.2.73",
		Port:        18435,
		Addresses:   []string{"192.0.2.73"},
		Models:      []string{exactCapability},
		ClusterUUID: "pinned-peer-principal",
	}

	if proxy.isManualCandidateEndpoint(
		subscribed,
		mustCapabilityURL(t, "http://192.0.2.73:18435"),
	) {
		t.Fatal("same-ID pinned subscribed endpoint was classified as manual")
	}

	t.Run("existing pinned mTLS routing remains non-local", func(t *testing.T) {
		TestResolveCandidatesRemoteMTLSRemainsNonLocal(t)
	})
}

func TestEndpointAwareSelfFacadeRewritesToLocalBackend(t *testing.T) {
	discovery := NewDiscovery()
	discovery.SetSubscribed([]Node{{
		ID:        "self",
		Host:      "127.0.0.1",
		Port:      18435,
		Addresses: []string{"127.0.0.1"},
		Models:    []string{compactCapability},
	}})
	discovery.AddManual(Node{
		ID:        "self",
		Host:      "127.0.0.1",
		Port:      18434,
		Addresses: []string{"127.0.0.1"},
		Models:    []string{exactCapability},
	})

	proxy := testProxy(discovery, 18435)
	proxy.setLocalBackend(localBackend{
		Engine:  "llamacpp",
		Host:    "127.0.0.1",
		Port:    18434,
		Healthy: true,
	})
	proxy.SetSelected("self")

	candidates := proxy.resolveCandidates(exactCapability)
	if len(candidates) != 1 {
		t.Fatalf("candidate count = %d, want 1", len(candidates))
	}
	if candidates[0].url.Host != "127.0.0.1:18434" {
		t.Fatalf("target = %q, want 127.0.0.1:18434", candidates[0].url.Host)
	}
	if !candidates[0].local {
		t.Fatal("rewritten exact local backend was not classified local")
	}
	if candidates[0].peerUUID != "" {
		t.Fatalf("local peer UUID = %q, want empty", candidates[0].peerUUID)
	}
}
