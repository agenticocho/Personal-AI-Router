// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"

	"nvpair-shared/noderec"
)

func directConnectManager(uuid string) *Manager {
	return &Manager{identity: &NodeIdentity{NodeUUID: uuid}}
}

func TestDirectConnectStateStartsEmpty(t *testing.T) {
	if got := directConnectManager("host-a").directConnectDescriptor(); got != nil {
		t.Fatalf("fresh process published a descriptor: %#v", got)
	}
}

func TestDirectConnectDerivesPrincipalFromLocalIdentity(t *testing.T) {
	m := directConnectManager("host-a")
	raw, _ := json.Marshal(noderec.DirectConnectParams{HostUUID: "host-a", Services: noderec.ServiceMap{noderec.ServiceLlamaCpp: 8081}})
	m.applyDirectConnect(&Message{Params: raw})
	desc := m.directConnectDescriptor()
	if desc == nil || desc.Principal != "host-a" || desc.HostUUID != "host-a" {
		t.Fatalf("descriptor=%#v", desc)
	}
	if desc.Services[noderec.ServiceLlamaCpp] != 8081 {
		t.Fatalf("port rewritten: %#v", desc.Services)
	}
}

func TestDirectConnectRejectsForeignAndMalformedSnapshots(t *testing.T) {
	m := directConnectManager("host-a")
	foreign, _ := json.Marshal(noderec.DirectConnectParams{HostUUID: "host-b", Services: noderec.ServiceMap{noderec.ServiceLlamaCpp: 8081}})
	m.applyDirectConnect(&Message{Params: foreign})
	if m.directConnectDescriptor() != nil {
		t.Fatal("foreign hostUuid accepted")
	}
	ok, _ := json.Marshal(noderec.DirectConnectParams{HostUUID: "host-a", Services: noderec.ServiceMap{noderec.ServiceLlamaCpp: 18435}})
	m.applyDirectConnect(&Message{Params: ok})
	if m.directConnectDescriptor() == nil {
		t.Fatal("valid snapshot rejected")
	}
	m.applyDirectConnect(&Message{Params: json.RawMessage(`{`)})
	if m.directConnectDescriptor() != nil {
		t.Fatal("malformed snapshot left a stale descriptor")
	}
}
