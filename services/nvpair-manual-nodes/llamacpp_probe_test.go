// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

// TestParseLlamaCppModelsSplitsLoadedState is the inventory contract: ids in
// the peer's order, the loaded subset identified from per-model status, and no
// attribution of another engine's models to llama.cpp.
func TestParseLlamaCppModelsSplitsLoadedState(t *testing.T) {
	body := []byte(`{"data":[
		{"id":"Ornith-1.5-9B-Q4_K_M","owned_by":"llamacpp","status":{"value":"loaded"}},
		{"id":"qwen2.5-3b-instruct-q4_k_m","owned_by":"llamacpp","status":{"value":"unloaded"}},
		{"id":"no-owner-declared","status":{"value":"loaded"}},
		{"id":"someone-elses","owned_by":"ollama","status":{"value":"loaded"}},
		{"id":"Ornith-1.5-9B-Q4_K_M","owned_by":"llamacpp","status":{"value":"loaded"}},
		{"id":"","owned_by":"llamacpp"}
	]}`)

	models, loaded := parseLlamaCppModels(body)

	wantModels := []string{"Ornith-1.5-9B-Q4_K_M", "qwen2.5-3b-instruct-q4_k_m", "no-owner-declared"}
	if !stringSliceEqual(models, wantModels) {
		t.Fatalf("models = %v, want %v", models, wantModels)
	}
	wantLoaded := []string{"Ornith-1.5-9B-Q4_K_M", "no-owner-declared"}
	if !stringSliceEqual(loaded, wantLoaded) {
		t.Fatalf("loaded = %v, want %v", loaded, wantLoaded)
	}
}

func TestParseLlamaCppModelsRejectsGarbage(t *testing.T) {
	for name, body := range map[string][]byte{
		"not json":    []byte("<html>nope</html>"),
		"empty":       []byte(""),
		"no data key": []byte(`{"object":"list"}`),
		"null data":   []byte(`{"data":null}`),
	} {
		models, loaded := parseLlamaCppModels(body)
		if len(models) != 0 || len(loaded) != 0 {
			t.Fatalf("%s: got models=%v loaded=%v, want none", name, models, loaded)
		}
	}
}

// TestProbeLlamaCppAuthenticatedRefusesWithoutPrincipalOrPort guards the
// preconditions: with no principal to select a pin for, or a port outside the
// valid range, the prober must not dial at all.
func TestProbeLlamaCppAuthenticatedRefusesWithoutPrincipalOrPort(t *testing.T) {
	var m Manager
	for _, tc := range []struct {
		name      string
		addr      string
		principal string
		port      int
	}{
		{"no principal", "10.0.0.9", "", 8081},
		{"no address", "", "host-a", 8081},
		{"port zero", "10.0.0.9", "host-a", 0},
		{"port too high", "10.0.0.9", "host-a", 70000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			up, models, loaded := m.probeLlamaCppAuthenticated(tc.addr, tc.principal, tc.port)
			if up || models != nil || loaded != nil {
				t.Fatalf("up=%v models=%v loaded=%v, want a refusal", up, models, loaded)
			}
		})
	}
}

func TestPrincipalFor(t *testing.T) {
	empty := ""
	value := "e692419e-02d3-472a-a93a-4e4f87ad652f"
	if got := principalFor(NodeInfoResponse{}); got != "" {
		t.Fatalf("absent clusterUuid: %q, want empty", got)
	}
	if got := principalFor(NodeInfoResponse{ClusterUUID: &empty}); got != "" {
		t.Fatalf("empty clusterUuid: %q, want empty", got)
	}
	if got := principalFor(NodeInfoResponse{ClusterUUID: &value}); got != value {
		t.Fatalf("clusterUuid: %q, want %q", got, value)
	}
}
