// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"log"
	"sync"

	"nvpair-shared/noderec"
)

// directConnectState holds this node's published direct-connect descriptor.
// It is deliberately memory-only: it starts empty on process start, is never
// read from or written to disk, never enters members.json or the pin store, and
// disappears on cluster-manager restart until the broker replays it.
type directConnectState struct {
	mu   sync.RWMutex
	desc *noderec.DirectConnect
}

func (s *directConnectState) set(d *noderec.DirectConnect) {
	s.mu.Lock()
	s.desc = d
	s.mu.Unlock()
}

func (s *directConnectState) get() *noderec.DirectConnect {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.desc == nil {
		return nil
	}
	clone := s.desc.Clone()
	return &clone
}

// applyDirectConnect ingests the broker's complete replacement snapshot. The
// broker supplies only hostUuid and services; the principal is derived here
// from this process's own authenticated identity, and both identity fields must
// equal it before the descriptor may ever be emitted. A malformed or
// mismatched snapshot clears the descriptor rather than latching a wrong one.
func (m *Manager) applyDirectConnect(msg *Message) {
	var params noderec.DirectConnectParams
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		log.Printf("ignoring malformed direct-connect snapshot: %v", err)
		m.directConnect.set(nil)
		return
	}
	if params.HostUUID != m.identity.NodeUUID {
		log.Printf("refusing direct-connect snapshot for foreign host %q", params.HostUUID)
		m.directConnect.set(nil)
		return
	}
	desc := noderec.DirectConnect{
		HostUUID:  m.identity.NodeUUID,
		Principal: m.identity.NodeUUID,
		Services:  noderec.SanitizeServices(params.Services),
	}
	if !desc.BoundTo(m.identity.NodeUUID) {
		m.directConnect.set(nil)
		return
	}
	m.directConnect.set(&desc)
	log.Printf("direct-connect descriptor updated (%d services)", len(desc.Services))
}

// directConnectDescriptor returns the descriptor for the authenticated roster
// response, or nil when this node has none to publish.
func (m *Manager) directConnectDescriptor() *noderec.DirectConnect {
	desc := m.directConnect.get()
	if desc == nil || !desc.BoundTo(m.identity.NodeUUID) {
		return nil
	}
	return desc
}
