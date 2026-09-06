// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"nvpair-shared/appdir"
	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
)

func redirectLocalAuthConfig(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	t.Setenv("HOME", root)
	t.Setenv("XDG_CONFIG_HOME", root)
	t.Setenv("APPDATA", root)
	t.Setenv("LOCALAPPDATA", root)

	path, err := appdir.Path("engines", "llamacpp.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeLocalAuthConfig(t *testing.T, configPath, tokenPath string) {
	t.Helper()
	data, err := json.Marshal(map[string]any{
		"engine": "llamacpp",
		"runtime": map[string]string{
			"bearer_token_file": tokenPath,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeLocalAuthToken(t *testing.T, path, token string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(token+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}

func candidateForServer(t *testing.T, serverURL string, local bool) candidate {
	t.Helper()
	parsed, err := url.Parse(serverURL)
	if err != nil {
		t.Fatal(err)
	}
	return candidate{id: "candidate", url: parsed, local: local}
}

func TestLocalBearerTokenFileMissingIsNoAuth(t *testing.T) {
	_ = redirectLocalAuthConfig(t)

	request := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/", nil)
	request.Header.Set("Authorization", "Bearer caller")
	if err := applyLocalBackendAuthorization(request); err != nil {
		t.Fatal(err)
	}
	if request.Header.Get("Authorization") != "Bearer caller" {
		t.Fatal("missing local auth configuration changed caller authorization")
	}
}

func TestLocalBearerTokenFileReloadsConfigAndToken(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	firstPath := filepath.Join(t.TempDir(), "first-token")
	secondPath := filepath.Join(t.TempDir(), "second-token")
	writeLocalAuthToken(t, firstPath, "first")
	writeLocalAuthToken(t, secondPath, "second")

	writeLocalAuthConfig(t, configPath, firstPath)
	first := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/", nil)
	if err := applyLocalBackendAuthorization(first); err != nil {
		t.Fatal(err)
	}
	if first.Header.Get("Authorization") != "Bearer first" {
		t.Fatal("first local request did not use first configured token")
	}

	writeLocalAuthConfig(t, configPath, secondPath)
	second := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/", nil)
	if err := applyLocalBackendAuthorization(second); err != nil {
		t.Fatal(err)
	}
	if second.Header.Get("Authorization") != "Bearer second" {
		t.Fatal("second local request did not reload configuration and token")
	}
}

func TestLocalBearerTokenFileFailureIsGeneric(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	sensitivePath := filepath.Join(t.TempDir(), "sensitive-token-name")
	writeLocalAuthConfig(t, configPath, sensitivePath)

	err := applyLocalBackendAuthorization(
		httptest.NewRequest(http.MethodGet, "http://127.0.0.1/", nil),
	)
	if err == nil {
		t.Fatal("expected local authentication failure")
	}
	for _, forbidden := range []string{
		configPath,
		sensitivePath,
		"sensitive-token-name",
	} {
		if strings.Contains(err.Error(), forbidden) {
			t.Fatal("local authentication error disclosed a path")
		}
	}
}

func TestServeModelListAppliesAuthOnlyToLocalCandidate(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "token")
	writeLocalAuthToken(t, tokenPath, "local-token")
	writeLocalAuthConfig(t, configPath, tokenPath)

	localServer := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer local-token" {
				t.Error("local model-list request lacked destination authorization")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			if r.Header.Get("Cookie") != "" {
				t.Error("caller cookie leaked into local model-list request")
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"local"}]}`))
		},
	))
	defer localServer.Close()

	remoteServer := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "" {
				t.Error("origin-local token leaked to non-local candidate")
			}
			if r.Header.Get("Cookie") != "" {
				t.Error("caller cookie leaked to non-local candidate")
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"remote"}]}`))
		},
	))
	defer remoteServer.Close()

	proxy := testProxy(NewDiscovery(), 18435)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	request.Header.Set("Authorization", "Bearer caller-secret")
	request.Header.Set("Cookie", "session=caller-secret")

	status, err := proxy.serveModelList(
		recorder,
		request,
		[]candidate{
			candidateForServer(t, localServer.URL, true),
			candidateForServer(t, remoteServer.URL, false),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200", status)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, `"local"`) ||
		!strings.Contains(body, `"remote"`) {
		t.Fatal("merged inventory did not contain both candidates")
	}
}

func TestServeModelListLocalAuthFailureIsGeneric(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	sensitivePath := filepath.Join(t.TempDir(), "private-token-name")
	writeLocalAuthConfig(t, configPath, sensitivePath)

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			t.Error("local backend was contacted after authentication failure")
		},
	))
	defer server.Close()

	proxy := testProxy(NewDiscovery(), 18435)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	status, err := proxy.serveModelList(
		recorder,
		request,
		[]candidate{candidateForServer(t, server.URL, true)},
	)
	if status != http.StatusServiceUnavailable || err == nil {
		t.Fatalf("status/error = %d/%v, want 503/error", status, err)
	}
	if recorder.Body.String() != `{"error":"model inventory unavailable"}` {
		t.Fatalf("unexpected public error body %q", recorder.Body.String())
	}
	for _, forbidden := range []string{
		configPath,
		sensitivePath,
		"private-token-name",
	} {
		if strings.Contains(err.Error(), forbidden) ||
			strings.Contains(recorder.Body.String(), forbidden) {
			t.Fatal("local authentication failure disclosed a path")
		}
	}
}

func TestResolveCandidatesMarksOnlySelfLocal(t *testing.T) {
	discovery := NewDiscovery()
	discovery.AddManual(Node{
		ID:        "self",
		Addresses: []string{"127.0.0.1"},
		Port:      18435,
	})
	discovery.AddManual(Node{
		ID:        "manual",
		Addresses: []string{"127.0.0.1"},
		Port:      19001,
	})

	proxy := testProxy(discovery, 18435)
	proxy.setLocalBackend(localBackend{
		Engine:  "llamacpp",
		Host:    "127.0.0.1",
		Port:    18434,
		Healthy: true,
	})

	candidates := proxy.resolveCandidates("")
	if len(candidates) != 2 {
		t.Fatalf("candidates = %v, want 2", candidates)
	}

	localCount := 0
	for _, candidate := range candidates {
		if candidate.local {
			localCount++
			if candidate.id != "self" {
				t.Fatalf("candidate %q incorrectly marked local", candidate.id)
			}
		}
		if candidate.id == "manual" && candidate.local {
			t.Fatal("manual candidate received local identity")
		}
	}
	if localCount != 1 {
		t.Fatalf("local candidate count = %d, want 1", localCount)
	}
}

func TestSameEndpointUsesStrictTextualIdentity(t *testing.T) {
	tests := []struct {
		name string
		a    string
		b    string
		want bool
	}{
		{
			name: "scheme case and terminal dot normalize",
			a:    "HTTP://Example.COM.:80",
			b:    "http://example.com",
			want: true,
		},
		{
			name: "https default port",
			a:    "https://example.com",
			b:    "HTTPS://EXAMPLE.COM:443",
			want: true,
		},
		{
			name: "same host different port",
			a:    "http://127.0.0.1:18434",
			b:    "http://127.0.0.1:18435",
			want: false,
		},
		{
			name: "different textual loopback host",
			a:    "http://127.0.0.1:18434",
			b:    "http://localhost:18434",
			want: false,
		},
		{
			name: "different loopback family",
			a:    "http://127.0.0.1:18434",
			b:    "http://[::1]:18434",
			want: false,
		},
		{
			name: "unknown scheme without ports",
			a:    "custom://example.com",
			b:    "CUSTOM://EXAMPLE.COM",
			want: false,
		},
		{
			name: "unknown scheme with explicit equal ports",
			a:    "custom://example.com:9000",
			b:    "CUSTOM://EXAMPLE.COM:9000",
			want: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a, err := url.Parse(test.a)
			if err != nil {
				t.Fatal(err)
			}
			b, err := url.Parse(test.b)
			if err != nil {
				t.Fatal(err)
			}

			if got := sameEndpoint(a, b); got != test.want {
				t.Fatalf(
					"sameEndpoint(%q, %q) = %t, want %t",
					test.a,
					test.b,
					got,
					test.want,
				)
			}
		})
	}
}

func directLocalRoutingProxy(t *testing.T, backendURL, model string) *Proxy {
	t.Helper()

	backend, err := url.Parse(backendURL)
	if err != nil {
		t.Fatal(err)
	}

	port, err := strconv.Atoi(backend.Port())
	if err != nil {
		t.Fatal(err)
	}

	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "direct-local", backendURL, model))

	proxy := testProxy(discovery, 18435)
	proxy.setLocalBackend(localBackend{
		Engine:  "llamacpp",
		Host:    backend.Hostname(),
		Port:    port,
		Healthy: true,
	})
	return proxy
}

func TestResolveCandidatesMarksExactDirectLocal(t *testing.T) {
	discovery := NewDiscovery()
	discovery.AddManual(Node{
		ID:        "direct-local",
		Addresses: []string{"127.0.0.1"},
		Port:      18434,
	})
	discovery.AddManual(Node{
		ID:        "same-host-different-port",
		Addresses: []string{"127.0.0.1"},
		Port:      19001,
	})
	discovery.AddManual(Node{
		ID:        "different-textual-host",
		Addresses: []string{"localhost"},
		Port:      18434,
	})

	proxy := testProxy(discovery, 18435)
	proxy.setLocalBackend(localBackend{
		Engine:  "llamacpp",
		Host:    "127.0.0.1",
		Port:    18434,
		Healthy: true,
	})

	candidates := proxy.resolveCandidates("")
	if len(candidates) != 3 {
		t.Fatalf("candidates = %#v, want 3", candidates)
	}

	for _, candidate := range candidates {
		switch candidate.id {
		case "direct-local":
			if !candidate.local {
				t.Fatal("exact direct local backend was not marked local")
			}
			if candidate.url.Scheme != "http" ||
				candidate.url.Host != "127.0.0.1:18434" {
				t.Fatalf(
					"direct local URL = %s, want authoritative http://127.0.0.1:18434",
					candidate.url.String(),
				)
			}
			if candidate.peerUUID != "" {
				t.Fatalf(
					"direct local peer UUID = %q, want empty",
					candidate.peerUUID,
				)
			}
		case "same-host-different-port", "different-textual-host":
			if candidate.local {
				t.Fatalf(
					"non-exact candidate %q was incorrectly marked local",
					candidate.id,
				)
			}
		default:
			t.Fatalf("unexpected candidate %q", candidate.id)
		}
	}
}

func TestResolveCandidatesRemoteMTLSRemainsNonLocal(t *testing.T) {
	const peerUUID = "principal-peer"
	clusterDir := filepath.Join(t.TempDir(), "cluster")

	discovery := NewDiscovery()
	discovery.SetSubscribed([]Node{{
		ID:          "remote-peer",
		Host:        "remote-peer",
		Port:        18435,
		Addresses:   []string{"192.0.2.10"},
		IP:          "192.0.2.10",
		ClusterUUID: peerUUID,
	}})

	proxy := testProxy(discovery, 18435)
	proxy.mesh = clustertrust.Open(clusterDir)
	clustertrusttest.Join(
		t,
		clusterDir,
		"cluster-xyz",
		"principal-self",
		peerUUID,
	)

	candidates := proxy.resolveCandidates("")
	if len(candidates) != 1 {
		t.Fatalf("candidates = %#v, want one pinned remote peer", candidates)
	}

	candidate := candidates[0]
	if candidate.id != "remote-peer" {
		t.Fatalf("candidate ID = %q, want remote-peer", candidate.id)
	}
	if candidate.local {
		t.Fatal("remote mTLS candidate was incorrectly marked local")
	}
	if candidate.peerUUID != peerUUID {
		t.Fatalf(
			"remote peer UUID = %q, want %q",
			candidate.peerUUID,
			peerUUID,
		)
	}
	if candidate.url.Scheme != "https" {
		t.Fatalf(
			"remote candidate scheme = %q, want https",
			candidate.url.Scheme,
		)
	}
}

func TestResolveCandidatesFacadeSelfWithoutBackendIsUnroutable(t *testing.T) {
	discovery := NewDiscovery()
	discovery.AddManual(Node{
		ID:        "self",
		Addresses: []string{"127.0.0.1"},
		Port:      18435,
	})

	proxy := testProxy(discovery, 18435)
	if candidates := proxy.resolveCandidates(""); len(candidates) != 0 {
		t.Fatalf(
			"candidates = %#v, want none without a healthy local backend",
			candidates,
		)
	}
}

func TestHandleHTTPDirectLocalModelListUsesDestinationAuth(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "token")
	writeLocalAuthToken(t, tokenPath, "destination-token")
	writeLocalAuthConfig(t, configPath, tokenPath)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer destination-token" {
			t.Error("direct local model-list request lacked destination authorization")
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if r.Header.Get("Cookie") != "" {
			t.Error("caller cookie reached direct local model-list backend")
		}
		if r.Header.Get("Connection") != "" {
			t.Error("Connection header reached direct local model-list backend")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(
			[]byte(`{"object":"list","data":[{"id":"direct-local"}]}`),
		)
	}))
	defer backend.Close()

	proxy := directLocalRoutingProxy(t, backend.URL, "direct-local-model")
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	request.Header.Set("Connection", "Authorization")
	request.Header.Set("Authorization", "Bearer caller-token")
	request.Header.Set("Cookie", "session=caller-cookie")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), "direct-local") {
		t.Fatalf(
			"model-list body = %q, want direct-local model",
			recorder.Body.String(),
		)
	}
}

func TestHandleHTTPDirectLocalInferenceUsesDestinationAuth(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "token")
	writeLocalAuthToken(t, tokenPath, "destination-token")
	writeLocalAuthConfig(t, configPath, tokenPath)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization := r.Header.Get("Authorization")
		if authorization != "Bearer destination-token" {
			t.Error("direct local inference lacked destination authorization")
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if strings.Contains(authorization, "caller-token") {
			t.Error("caller authorization survived direct local replacement")
		}
		if r.Header.Get("Cookie") != "" {
			t.Error("caller cookie reached direct local inference backend")
		}
		if r.Header.Get("Connection") != "" {
			t.Error("Connection header reached direct local inference backend")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(
			[]byte(`{"choices":[{"message":{"content":"ok"}}]}`),
		)
	}))
	defer backend.Close()

	proxy := directLocalRoutingProxy(t, backend.URL, "direct-local-model")
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"direct-local-model"}`),
	)
	request.Header.Set("Connection", "Authorization")
	request.Header.Set("Authorization", "Bearer caller-token")
	request.Header.Set("Cookie", "session=caller-cookie")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
}

func TestLocalBearerTokenFileRejectsInsecureToken(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix permission test")
	}

	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "token")
	writeLocalAuthToken(t, tokenPath, "local-token")
	if err := os.Chmod(tokenPath, 0o644); err != nil {
		t.Fatal(err)
	}
	writeLocalAuthConfig(t, configPath, tokenPath)

	request := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/", nil)
	err := applyLocalBackendAuthorization(request)
	if err == nil {
		t.Fatal("expected insecure local token rejection")
	}
	if request.Header.Get("Authorization") != "" {
		t.Fatal("insecure local token modified Authorization")
	}
}

func selfRoutingProxy(
	t *testing.T,
	backendURL string,
	model string,
) *Proxy {
	t.Helper()

	backend, err := url.Parse(backendURL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(backend.Port())
	if err != nil {
		t.Fatal(err)
	}

	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(
		t,
		"self",
		"http://127.0.0.1:18435",
		model,
	))
	proxy := testProxy(discovery, 18435)
	proxy.setLocalBackend(localBackend{
		Engine:  "llamacpp",
		Host:    backend.Hostname(),
		Port:    port,
		Healthy: true,
	})
	return proxy
}

func TestHandleHTTPLocalInferenceUsesDestinationAuth(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "token")
	writeLocalAuthToken(t, tokenPath, "destination-token")
	writeLocalAuthConfig(t, configPath, tokenPath)

	backend := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			authorization := r.Header.Get("Authorization")
			if authorization != "Bearer destination-token" {
				t.Error("local inference did not receive destination authorization")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			if strings.Contains(authorization, "caller-token") {
				t.Error("caller authorization survived local replacement")
			}
			if r.Header.Get("Connection") != "" {
				t.Error("Connection header reached local inference backend")
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
		},
	))
	defer backend.Close()

	proxy := selfRoutingProxy(t, backend.URL, "model")
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"model"}`),
	)
	request.Header.Set("Connection", "Authorization")
	request.Header.Set("Authorization", "Bearer caller-token")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
}

func TestHandleHTTPManualInferenceNeverGetsLocalToken(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "token")
	writeLocalAuthToken(t, tokenPath, "destination-token")
	writeLocalAuthConfig(t, configPath, tokenPath)

	manual := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer caller-token" {
				t.Error("manual candidate did not preserve caller authorization")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			w.WriteHeader(http.StatusOK)
		},
	))
	defer manual.Close()

	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "manual", manual.URL, "model"))
	proxy := testProxy(discovery, 18435)

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"model"}`),
	)
	request.Header.Set("Authorization", "Bearer caller-token")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
}

func TestHandleHTTPLocalTokenDoesNotLeakAfterStatusFailover(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "token")
	writeLocalAuthToken(t, tokenPath, "destination-token")
	writeLocalAuthConfig(t, configPath, tokenPath)

	local := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer destination-token" {
				t.Error("local candidate lacked destination token")
			}
			http.Error(w, "busy", http.StatusServiceUnavailable)
		},
	))
	defer local.Close()

	remote := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer caller-token" {
				t.Error("local token leaked into remote failover request")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			w.WriteHeader(http.StatusOK)
		},
	))
	defer remote.Close()

	localURL, err := url.Parse(local.URL)
	if err != nil {
		t.Fatal(err)
	}
	localPort, err := strconv.Atoi(localURL.Port())
	if err != nil {
		t.Fatal(err)
	}

	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(
		t,
		"self",
		"http://127.0.0.1:18435",
		"model",
	))
	discovery.AddManual(nodeForModel(t, "remote", remote.URL, "model"))

	proxy := testProxy(discovery, 18435)
	proxy.setLocalBackend(localBackend{
		Engine:  "llamacpp",
		Host:    localURL.Hostname(),
		Port:    localPort,
		Healthy: true,
	})
	proxy.SetSelected("self")

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"model"}`),
	)
	request.Header.Set("Authorization", "Bearer caller-token")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want remote failover 200", recorder.Code)
	}
}

func TestHandleHTTPLocalAuthFailureFailsOverWithoutDisclosure(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	sensitivePath := filepath.Join(t.TempDir(), "private-token-name")
	writeLocalAuthConfig(t, configPath, sensitivePath)

	localHits := 0
	local := httptest.NewServer(http.HandlerFunc(
		func(http.ResponseWriter, *http.Request) {
			localHits++
		},
	))
	defer local.Close()

	remote := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer caller-token" {
				t.Error("auth-failure failover request was contaminated")
			}
			w.WriteHeader(http.StatusOK)
		},
	))
	defer remote.Close()

	localURL, err := url.Parse(local.URL)
	if err != nil {
		t.Fatal(err)
	}
	localPort, err := strconv.Atoi(localURL.Port())
	if err != nil {
		t.Fatal(err)
	}

	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(
		t,
		"self",
		"http://127.0.0.1:18435",
		"model",
	))
	discovery.AddManual(nodeForModel(t, "remote", remote.URL, "model"))

	proxy := testProxy(discovery, 18435)
	proxy.setLocalBackend(localBackend{
		Engine:  "llamacpp",
		Host:    localURL.Hostname(),
		Port:    localPort,
		Healthy: true,
	})
	proxy.SetSelected("self")

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"model"}`),
	)
	request.Header.Set("Authorization", "Bearer caller-token")
	recorder := httptest.NewRecorder()

	proxy.handleHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want failover 200", recorder.Code)
	}
	if localHits != 0 {
		t.Fatal("local backend contacted despite credential failure")
	}
	for _, forbidden := range []string{
		configPath,
		sensitivePath,
		"private-token-name",
	} {
		if strings.Contains(recorder.Body.String(), forbidden) {
			t.Fatal("auth-failure response disclosed a path")
		}
	}
}

func TestReverseProxyToLocalOverwritesCallerAuthorization(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	tokenPath := filepath.Join(t.TempDir(), "token")
	writeLocalAuthToken(t, tokenPath, "destination-token")
	writeLocalAuthConfig(t, configPath, tokenPath)

	backend := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			authorization := r.Header.Get("Authorization")
			if authorization != "Bearer destination-token" {
				t.Error("ingress hop did not overwrite caller authorization")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			if strings.Contains(authorization, "peer-token") {
				t.Error("peer authorization survived local replacement")
			}
			if r.Header.Get("Connection") != "" {
				t.Error("Connection header reached ingress backend")
			}
			w.WriteHeader(http.StatusOK)
		},
	))
	defer backend.Close()

	target, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	proxy := testProxy(NewDiscovery(), 18435)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"model"}`),
	)
	request.Header.Set("Connection", "Authorization")
	request.Header.Set("Authorization", "Bearer peer-token")
	recorder := httptest.NewRecorder()

	proxy.reverseProxyToLocal(recorder, request, target)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
}

func TestReverseProxyToLocalAuthFailureIsGeneric(t *testing.T) {
	configPath := redirectLocalAuthConfig(t)
	sensitivePath := filepath.Join(t.TempDir(), "private-token-name")
	writeLocalAuthConfig(t, configPath, sensitivePath)

	hits := 0
	backend := httptest.NewServer(http.HandlerFunc(
		func(http.ResponseWriter, *http.Request) {
			hits++
		},
	))
	defer backend.Close()

	target, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	proxy := testProxy(NewDiscovery(), 18435)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	proxy.reverseProxyToLocal(recorder, request, target)
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", recorder.Code)
	}
	if hits != 0 {
		t.Fatal("backend contacted after local auth failure")
	}
	for _, forbidden := range []string{
		configPath,
		sensitivePath,
		"private-token-name",
	} {
		if strings.Contains(recorder.Body.String(), forbidden) {
			t.Fatal("ingress auth error disclosed a path")
		}
	}
}

func TestReverseProxyToLocalNoAuthPreservesCallerAuthorization(t *testing.T) {
	_ = redirectLocalAuthConfig(t)

	backend := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer peer-token" {
				t.Error("no-auth configuration changed caller authorization")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			w.WriteHeader(http.StatusOK)
		},
	))
	defer backend.Close()

	target, err := url.Parse(backend.URL)
	if err != nil {
		t.Fatal(err)
	}

	proxy := testProxy(NewDiscovery(), 18435)
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	request.Header.Set("Authorization", "Bearer peer-token")
	recorder := httptest.NewRecorder()

	proxy.reverseProxyToLocal(recorder, request, target)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
}
