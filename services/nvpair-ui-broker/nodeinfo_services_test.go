// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

type replayBuffer struct{ bytes.Buffer }

func (*replayBuffer) Close() error { return nil }
func decodeServiceReplay(t *testing.T, b []byte) noderec.ServiceMap {
	t.Helper()
	var frame struct {
		Method string                   `json:"method"`
		Params noderec.ServiceMapParams `json:"params"`
	}
	if err := json.Unmarshal(bytes.TrimSpace(b), &frame); err != nil {
		t.Fatal(err)
	}
	if frame.Method != noderec.MethodSetServices {
		t.Fatalf("method=%q", frame.Method)
	}
	return frame.Params.Services
}
func TestNodeInfoServiceSnapshotReplayAcrossReplacement(t *testing.T) {
	b := &Broker{regCache: relay.NewRegistrationCache()}
	b.regCache.Register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 14318})
	b.regCache.Register(noderec.RegisterParams{Service: noderec.ServiceLlamaCpp, Port: 18435})
	first := &replayBuffer{}
	b.setNodeInfo(&nodeInfoProcess{stdin: first})
	b.pushServicesToNodeInfo()
	one := decodeServiceReplay(t, first.Bytes())
	second := &replayBuffer{}
	b.setNodeInfo(&nodeInfoProcess{stdin: second})
	b.pushServicesToNodeInfo()
	two := decodeServiceReplay(t, second.Bytes())
	for _, got := range []noderec.ServiceMap{one, two} {
		if got[noderec.ServiceNodeInfo] != 14318 || got[noderec.ServiceLlamaCpp] != 18435 || len(got) != 2 {
			t.Fatalf("incomplete replay: %#v", got)
		}
	}
}
