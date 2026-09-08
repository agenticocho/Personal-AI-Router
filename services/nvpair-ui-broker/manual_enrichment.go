// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// manualClusterUUID returns the cluster principal a manual node's probe
// reported, or "" when it reported none. Both projections read it through this
// one helper so they cannot disagree about what the peer claimed.
func manualClusterUUID(s manualNodeStatus) string {
	if s.ClusterUUID == nil {
		return ""
	}
	return *s.ClusterUUID
}

// manualTrusted reports whether a manual claim may carry the trusted
// annotation. It requires BOTH a principal and a valid authenticated service
// map: a principal alone is only what the peer said about itself over
// plaintext, while ServiceMapValid is this node's own record that the
// pinned-mTLS bootstrap accepted the descriptor. Anything less is an
// unauthenticated claim and must not be annotated as a paired peer.
func manualTrusted(s manualNodeStatus) bool {
	return s.ServiceMapValid && manualClusterUUID(s) != ""
}

// manualLoadedByEngine builds the per-engine resident-model attribution for a
// manual node, keyed by the same engine-manager engine names discovered nodes
// use. An engine reporting nothing resident adds no key; nil when no engine
// reported any, which reads as "no peer told us" rather than "nothing loaded".
//
// Only llama.cpp reports residency on the manual path today: the Ollama and LM
// Studio probes return an inventory without loaded state, so claiming an empty
// list for them would assert something the probe never observed.
func manualLoadedByEngine(s manualNodeStatus) map[string][]string {
	loaded := map[string][]string{}
	if len(s.LlamaCppLoaded) > 0 {
		loaded[llamaCppEngineName] = s.LlamaCppLoaded
	}
	if len(loaded) == 0 {
		return nil
	}
	return loaded
}

// llamaCppEngineName is the engine-manager engine name llama.cpp models are
// attributed to, matching the key the scanner-sourced enrichment uses so
// DirectoryNode.EngineModels resolves identically for either discovery source.
const llamaCppEngineName = "llamacpp"
