// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
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
)

func writePrivateTestToken(t *testing.T, token string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(path, []byte(token+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func testExecutorAtServer(
	t *testing.T,
	manifest *Manifest,
	serverURL string,
) *Executor {
	t.Helper()

	parsed, err := url.Parse(serverURL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatal(err)
	}

	platform := manifest.Platforms[hostKey()]
	platform.Runtime.Port = port
	manifest.Platforms[hostKey()] = platform

	executor := newTestExecutor(t, manifest)
	state, err := executor.state(manifest.Engine)
	if err != nil {
		t.Fatal(err)
	}
	state.mu.Lock()
	state.running = true
	state.port = port
	state.mu.Unlock()
	return executor
}

func TestActionUsesRuntimeBearerTokenFile(t *testing.T) {
	const token = "test-secret-token"
	tokenFile := writePrivateTestToken(t, token)

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer "+token {
				t.Error("request did not receive expected bearer authorization")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			if r.Header.Get(engineIdentityProbeHeader) != "1" {
				t.Error("request did not receive identity-probe header")
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"model"}]}`))
		},
	))
	defer server.Close()

	manifest := testEngineManifest(fakeEngineBin)
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.BearerTokenFile = tokenFile
	manifest.Platforms[hostKey()] = platform
	manifest.Actions["list_models"] = Action{
		HTTP: &ActionHTTP{
			Method: http.MethodGet,
			Path:   "/v1/models",
		},
	}

	executor := testExecutorAtServer(t, manifest, server.URL)
	result, err := executor.Action(
		context.Background(),
		manifest.Engine,
		"list_models",
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(result), `"model"`) {
		t.Fatalf("result did not contain expected model")
	}
}

func TestActionRuntimeBearerTokenFileRejectsInsecureMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix permission test")
	}

	tokenFile := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(tokenFile, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(tokenFile, 0o644); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/", nil)
	err := applyActionHTTPAuthorization(request, tokenFile)
	if err == nil || !strings.Contains(err.Error(), "group or other") {
		t.Fatalf("expected insecure-permission rejection, got %v", err)
	}
	if strings.Contains(err.Error(), "secret") {
		t.Fatal("error disclosed token")
	}
}

func TestActionRuntimeBearerTokenFileOmitted(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/", nil)

	if err := applyActionHTTPAuthorization(request, ""); err != nil {
		t.Fatal(err)
	}
	if request.Header.Get("Authorization") != "" {
		t.Fatal("Authorization unexpectedly set")
	}

	encoded, err := json.Marshal(Runtime{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "bearer_token_file") {
		t.Fatal("empty Runtime encoded credential field")
	}
}

func TestPullModelUsesRuntimeBearerTokenFile(t *testing.T) {
	const token = "test-pull-token"
	tokenFile := writePrivateTestToken(t, token)

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer "+token {
				t.Error("pull request did not receive expected bearer authorization")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte("{\"status\":\"success\"}\n"))
		},
	))
	defer server.Close()

	manifest := testEngineManifest(fakeEngineBin)
	platform := manifest.Platforms[hostKey()]
	platform.Runtime.BearerTokenFile = tokenFile
	manifest.Platforms[hostKey()] = platform
	manifest.Actions[pullModelAction] = Action{
		HTTP: &ActionHTTP{
			Method: http.MethodPost,
			Path:   "/api/pull",
		},
	}

	executor := testExecutorAtServer(t, manifest, server.URL)
	result, err := executor.PullModelStream(
		context.Background(),
		manifest.Engine,
		"demo:1b",
		json.RawMessage(`{"name":"demo:1b"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(result), `"success"`) {
		t.Fatal("pull result did not contain success")
	}
}

func TestActionRuntimeBearerTokenFileRejectsInvalidFiles(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    string
	}{
		{name: "empty", content: "", want: "is empty"},
		{name: "multiline", content: "first\nsecond\n", want: "multiple lines"},
		{
			name:    "oversized",
			content: strings.Repeat("x", maxBearerTokenFileBytes+1),
			want:    "exceeds",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tokenFile := filepath.Join(t.TempDir(), "token")
			if err := os.WriteFile(tokenFile, []byte(test.content), 0o600); err != nil {
				t.Fatal(err)
			}

			request := httptest.NewRequest(
				http.MethodGet,
				"http://127.0.0.1/",
				nil,
			)
			err := applyActionHTTPAuthorization(request, tokenFile)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q rejection, got %v", test.want, err)
			}
			if request.Header.Get("Authorization") != "" {
				t.Fatal("Authorization set after rejected credential file")
			}
		})
	}

	t.Run("directory", func(t *testing.T) {
		request := httptest.NewRequest(
			http.MethodGet,
			"http://127.0.0.1/",
			nil,
		)
		err := applyActionHTTPAuthorization(request, t.TempDir())
		if err == nil || !strings.Contains(err.Error(), "not a regular file") {
			t.Fatalf("expected non-regular-file rejection, got %v", err)
		}
		if request.Header.Get("Authorization") != "" {
			t.Fatal("Authorization set after rejected directory")
		}
	})
}

func TestRuntimeBearerTokenFileMinimalOverrideAuthenticatesAllHTTPActions(
	t *testing.T,
) {
	const token = "test-shared-runtime-token"
	home := t.TempDir()
	t.Setenv("HOME", home)

	tokenFile := filepath.Join(home, "runtime-token")
	if err := os.WriteFile(tokenFile, []byte(token+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	overrideDir := t.TempDir()
	override := []byte(`{
  "engine": "llamacpp",
  "runtime": {
    "bearer_token_file": "$HOME/runtime-token"
  }
}`)
	if err := os.WriteFile(
		filepath.Join(overrideDir, "llamacpp.json"),
		override,
		0o600,
	); err != nil {
		t.Fatal(err)
	}

	registry := loadWithOverrides(t, overrideDir)
	manifest, ok := registry.Get("llamacpp")
	if !ok {
		t.Fatal("llamacpp missing after override merge")
	}

	platform, ok := manifest.HostPlatform()
	if !ok {
		t.Fatal("llamacpp host platform missing after override merge")
	}
	if platform.Runtime.BearerTokenFile != "$HOME/runtime-token" {
		t.Fatal("runtime bearer token file was not preserved by deep merge")
	}

	for _, action := range []string{"list_models", "loaded_models"} {
		spec, ok := manifest.Actions[action]
		if !ok || spec.HTTP == nil {
			t.Fatalf("action %q was not preserved by minimal override", action)
		}
	}

	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			requestCount++
			if r.Header.Get("Authorization") != "Bearer "+token {
				t.Error("shared runtime authorization was not applied")
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"model"}]}`))
		},
	))
	defer server.Close()

	executor := testExecutorAtServer(t, manifest, server.URL)
	for _, action := range []string{"list_models", "loaded_models"} {
		if _, err := executor.Action(
			context.Background(),
			manifest.Engine,
			action,
			nil,
		); err != nil {
			t.Fatalf("action %q failed: %v", action, err)
		}
	}
	if requestCount != 2 {
		t.Fatalf("authenticated request count = %d, want 2", requestCount)
	}
}
