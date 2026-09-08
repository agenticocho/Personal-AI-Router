// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"nvpair-shared/noderec"
)

func acceptedLlamaCppStatus() manualNodeStatus {
	cluster := "34b0b4dd-1e10-4fd1-884f-4c3cc5dc498c"
	return manualNodeStatus{
		ID:              "chilly",
		Address:         "100.87.70.89",
		NodeInfoPort:    14318,
		HostUUID:        "34b0b4dd-1e10-4fd1-884f-4c3cc5dc498c",
		ClusterUUID:     &cluster,
		ServiceMapValid: true,
		Services: noderec.ServiceMap{
			noderec.ServiceLlamaCpp:      8081,
			noderec.ServiceEngineManager: 14322,
			noderec.ServiceEngineControl: 14323,
		},
		LlamaCppUp:     true,
		LlamaCppPort:   8081,
		LlamaCppModels: []string{"Ornith-1.5-9B-Q4_K_M", "qwen2.5-3b-instruct-q4_k_m"},
		LlamaCppLoaded: []string{"Ornith-1.5-9B-Q4_K_M"},
	}
}

// TestManualClaimCarriesLlamaCppInventory is the regression for the deployed
// 502: a llama.cpp-only peer whose descriptor was accepted must reach the
// directory with per-engine attribution, or routing eligibility finds no owner
// for its models even though listing works.
func TestManualClaimCarriesLlamaCppInventory(t *testing.T) {
	s := acceptedLlamaCppStatus()

	n, ok := manualToDirectoryNode(s)
	if !ok {
		t.Fatal("an accepted manual claim produced no directory node")
	}
	if !n.Trusted {
		t.Fatal("accepted claim is not annotated trusted")
	}
	if !n.Clustered() {
		t.Fatal("accepted claim does not report clustered")
	}
	if got := n.EngineModels(llamaCppEngineName); !equalStrings(got, s.LlamaCppModels) {
		t.Fatalf("EngineModels(llamacpp) = %v, want %v", got, s.LlamaCppModels)
	}
	if got := n.LoadedByEngine[llamaCppEngineName]; !equalStrings(got, s.LlamaCppLoaded) {
		t.Fatalf("LoadedByEngine[llamacpp] = %v, want %v", got, s.LlamaCppLoaded)
	}
	for _, want := range s.LlamaCppModels {
		if !contains(n.Models, want) {
			t.Fatalf("flat Models %v is missing %q", n.Models, want)
		}
	}
	if port := n.Services[noderec.ServiceLlamaCpp].Port; port != 8081 {
		t.Fatalf("ll port %d, want 8081 verbatim", port)
	}
}

// TestBothProjectionsAgree pins the invariant whose absence caused the second
// half of this defect: the relay claim and the client-facing claim must report
// the same trust and the same inventory for the same input.
func TestBothProjectionsAgree(t *testing.T) {
	s := acceptedLlamaCppStatus()

	n, ok := manualToDirectoryNode(s)
	if !ok {
		t.Fatal("no directory node")
	}
	en := manualToEnriched(s)

	if en.Trusted != n.Trusted {
		t.Fatalf("Trusted: enriched=%v directory=%v", en.Trusted, n.Trusted)
	}
	if en.Clustered != n.Clustered() {
		t.Fatalf("Clustered: enriched=%v directory=%v", en.Clustered, n.Clustered())
	}
	if !equalStrings(en.ModelsByEngine[llamaCppEngineName], n.ModelsByEngine[llamaCppEngineName]) {
		t.Fatalf("ModelsByEngine diverged: enriched=%v directory=%v",
			en.ModelsByEngine, n.ModelsByEngine)
	}
	if !equalStrings(en.LoadedByEngine[llamaCppEngineName], n.LoadedByEngine[llamaCppEngineName]) {
		t.Fatalf("LoadedByEngine diverged: enriched=%v directory=%v",
			en.LoadedByEngine, n.LoadedByEngine)
	}
	if !equalStrings(en.Models, n.Models) {
		t.Fatalf("flat Models diverged: enriched=%v directory=%v", en.Models, n.Models)
	}
}

// TestUnauthenticatedManualClaimIsNotTrustedOrEnriched is the negative case.
// An unauthenticated or failed probe must produce no directory node at all and
// must not annotate trust, whatever the peer said about itself over plaintext.
func TestUnauthenticatedManualClaimIsNotTrustedOrEnriched(t *testing.T) {
	t.Run("service map invalid", func(t *testing.T) {
		s := acceptedLlamaCppStatus()
		s.ServiceMapValid = false
		if _, ok := manualToDirectoryNode(s); ok {
			t.Fatal("an unauthenticated claim produced a directory node")
		}
		if en := manualToEnriched(s); en.Trusted {
			t.Fatal("an unauthenticated claim was annotated trusted")
		}
	})

	t.Run("no principal", func(t *testing.T) {
		s := acceptedLlamaCppStatus()
		s.ClusterUUID = nil
		if en := manualToEnriched(s); en.Trusted {
			t.Fatal("a claim with no principal was annotated trusted")
		}
		if _, ok := manualToDirectoryNode(s); ok {
			n, _ := manualToDirectoryNode(s)
			if n.Trusted {
				t.Fatal("a claim with no principal was annotated trusted in the directory")
			}
		}
	})

	t.Run("empty principal", func(t *testing.T) {
		s := acceptedLlamaCppStatus()
		empty := ""
		s.ClusterUUID = &empty
		if manualTrusted(s) {
			t.Fatal("an empty principal was treated as trust")
		}
	})
}

// TestManualLoadedByEngineOmitsSilentEngines: an engine that reported no
// resident models adds no key, so a consumer can tell "reported nothing
// loaded" from "was never asked".
func TestManualLoadedByEngineOmitsSilentEngines(t *testing.T) {
	s := acceptedLlamaCppStatus()
	s.LlamaCppLoaded = nil
	if got := manualLoadedByEngine(s); got != nil {
		t.Fatalf("LoadedByEngine = %v, want nil when nothing was reported", got)
	}
}

func equalStrings(a, b []string) bool {
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

func contains(haystack []string, needle string) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}
