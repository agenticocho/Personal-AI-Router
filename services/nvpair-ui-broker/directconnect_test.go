// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
)

func TestDirectConnectSnapshotOrderedRegistrationAndWithdrawal(t *testing.T) {
	cache := relay.NewRegistrationCache()
	cache.Register(noderec.RegisterParams{Service: noderec.ServiceNodeInfo, Port: 14318})
	cache.Register(noderec.RegisterParams{Service: noderec.ServiceLlamaCpp, Port: 8081})
	cache.Register(noderec.RegisterParams{Service: noderec.ServiceEngineControl, Port: 14322})

	first := directConnectSnapshot("host-a", cache.Snapshot())
	if first.HostUUID != "host-a" || len(first.Services) != 3 || first.Services[noderec.ServiceLlamaCpp] != 8081 {
		t.Fatalf("registration snapshot=%#v", first)
	}

	cache.Unregister(noderec.ServiceLlamaCpp)
	afterWithdrawal := directConnectSnapshot("host-a", cache.Snapshot())
	if _, ok := afterWithdrawal.Services[noderec.ServiceLlamaCpp]; ok {
		t.Fatalf("withdrawal not reflected: %#v", afterWithdrawal.Services)
	}
	if len(afterWithdrawal.Services) != 2 {
		t.Fatalf("withdrawal snapshot=%#v", afterWithdrawal.Services)
	}
	if len(first.Services) != 3 {
		t.Fatal("earlier snapshot mutated by a later withdrawal")
	}
}

func TestDirectConnectSnapshotRestartReplayIsComplete(t *testing.T) {
	cache := relay.NewRegistrationCache()
	for _, p := range []noderec.RegisterParams{
		{Service: noderec.ServiceNodeInfo, Port: 14318},
		{Service: noderec.ServiceCluster, Port: 14321},
		{Service: noderec.ServiceLlamaCpp, Port: 18435},
		{Service: noderec.ServiceEngineManager, Port: 14319},
	} {
		cache.Register(p)
	}
	replay := directConnectSnapshot("host-b", cache.Snapshot())
	if len(replay.Services) != 4 || replay.Services[noderec.ServiceLlamaCpp] != 18435 {
		t.Fatalf("replay is not the complete current snapshot: %#v", replay.Services)
	}
	again := directConnectSnapshot("host-b", cache.Snapshot())
	if len(again.Services) != len(replay.Services) {
		t.Fatal("replay is not idempotent")
	}
}

func TestDirectConnectSnapshotDropsNonPublicAndInvalid(t *testing.T) {
	got := directConnectSnapshot("host-a", []noderec.RegisterParams{
		{Service: noderec.ServiceLlamaCpp, Port: 8081},
		{Service: noderec.ServiceKey("zz"), Port: 9000},
		{Service: noderec.ServiceOllama, Port: 0},
		{Service: noderec.ServiceLMStudio, Port: 70000},
	})
	if len(got.Services) != 1 || got.Services[noderec.ServiceLlamaCpp] != 8081 {
		t.Fatalf("snapshot=%#v", got.Services)
	}
}

func TestDirectConnectStaleSnapshotIsCoalesced(t *testing.T) {
	stale := directConnectSeq.Add(1)
	newest := directConnectSeq.Add(1)
	if directConnectShouldSend(stale) {
		t.Fatal("stale snapshot would be sent after a newer one")
	}
	if !directConnectShouldSend(newest) {
		t.Fatal("newest snapshot was suppressed")
	}
}
