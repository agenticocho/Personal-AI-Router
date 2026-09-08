// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package relay owns the broker discovery star: local registration replay upward to the scanner and source-aware directory fanout downward to consumers.
package relay

import (
	"nvpair-shared/noderec"
	"reflect"
	"sort"
	"sync"
)

// RegistrationCache holds this node's current service registrations for deterministic scanner and node-info replay.
type RegistrationCache struct {
	mu   sync.Mutex
	regs map[noderec.ServiceKey]noderec.RegisterParams
}

// NewRegistrationCache returns an empty registration cache.
func NewRegistrationCache() *RegistrationCache {
	return &RegistrationCache{regs: make(map[noderec.ServiceKey]noderec.RegisterParams)}
}

// Register adds or replaces one valid service registration and reports whether the cache changed.
func (c *RegistrationCache) Register(p noderec.RegisterParams) bool {
	if p.Service == "" || p.Port == 0 {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if prev, ok := c.regs[p.Service]; ok && registerEqual(prev, p) {
		return false
	}
	c.regs[p.Service] = p
	return true
}

// Unregister withdraws one service and reports whether it existed.
func (c *RegistrationCache) Unregister(k noderec.ServiceKey) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.regs[k]; !ok {
		return false
	}
	delete(c.regs, k)
	return true
}

// Snapshot returns registrations sorted by service key.
func (c *RegistrationCache) Snapshot() []noderec.RegisterParams {
	c.mu.Lock()
	out := make([]noderec.RegisterParams, 0, len(c.regs))
	for _, p := range c.regs {
		out = append(out, p)
	}
	c.mu.Unlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Service < out[j].Service })
	return out
}
func registerEqual(a, b noderec.RegisterParams) bool {
	if a.Service != b.Service || a.Port != b.Port || len(a.TXT) != len(b.TXT) {
		return false
	}
	for i := range a.TXT {
		if a.TXT[i] != b.TXT[i] {
			return false
		}
	}
	return true
}

// Subscriber receives authoritative filtered directory snapshots.
type Subscriber struct {
	Filter noderec.SubscribeParams
	Send   func([]noderec.DirectoryNode)
	sendMu sync.Mutex
}
type nodeClaims struct {
	scanner *noderec.DirectoryNode
	manual  *noderec.DirectoryNode
}

// Directory merges scanner and authenticated manual claims by stable host UUID.
type Directory struct {
	mu     sync.Mutex
	nodes  map[string]nodeClaims
	subs   map[int]*Subscriber
	nextID int
}

// NewDirectory returns an empty source-aware directory.
func NewDirectory() *Directory {
	return &Directory{nodes: make(map[string]nodeClaims), subs: make(map[int]*Subscriber)}
}

// Subscribe registers a subscriber; callers invoke Deliver for its initial snapshot.
func (d *Directory) Subscribe(s *Subscriber) (id int) {
	d.mu.Lock()
	d.nextID++
	id = d.nextID
	d.subs[id] = s
	d.mu.Unlock()
	return
}

// Unsubscribe removes a subscriber.
func (d *Directory) Unsubscribe(id int) { d.mu.Lock(); delete(d.subs, id); d.mu.Unlock() }

// Deliver serializes and sends the subscriber's latest full filtered snapshot.
func (d *Directory) Deliver(s *Subscriber) {
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	d.mu.Lock()
	nodes := d.filteredLocked(s.Filter)
	d.mu.Unlock()
	s.Send(nodes)
}
func (d *Directory) filteredLocked(f noderec.SubscribeParams) []noderec.DirectoryNode {
	out := make([]noderec.DirectoryNode, 0, len(d.nodes))
	for _, c := range d.nodes {
		n, ok := project(c)
		if ok && f.Matches(n) {
			out = append(out, n)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].HostUUID < out[j].HostUUID })
	return out
}
func cloneStringMap(in map[string][]string) map[string][]string {
	if in == nil {
		return nil
	}
	out := make(map[string][]string, len(in))
	for key, values := range in {
		out[key] = append([]string(nil), values...)
	}
	return out
}

func appendUnique(dst []string, values ...[]string) []string {
	seen := make(map[string]bool, len(dst))
	for _, value := range dst {
		seen[value] = true
	}
	for _, list := range values {
		for _, value := range list {
			if value != "" && !seen[value] {
				seen[value] = true
				dst = append(dst, value)
			}
		}
	}
	return dst
}

func cloneNode(n noderec.DirectoryNode) noderec.DirectoryNode {
	out := n
	out.IPs = append([]string(nil), n.IPs...)
	out.GPUs = append([]noderec.GPUInfo(nil), n.GPUs...)
	out.Models = append([]string(nil), n.Models...)
	out.ModelsByEngine = cloneStringMap(n.ModelsByEngine)
	out.LoadedByEngine = cloneStringMap(n.LoadedByEngine)
	if n.CPU != nil {
		value := *n.CPU
		out.CPU = &value
	}
	if n.Memory != nil {
		value := *n.Memory
		out.Memory = &value
	}
	out.Services = make(map[noderec.ServiceKey]noderec.ServiceStatus, len(n.Services))
	for k, v := range n.Services {
		out.Services[k] = v
	}
	return out
}
func project(c nodeClaims) (noderec.DirectoryNode, bool) {
	if c.scanner == nil && c.manual == nil {
		return noderec.DirectoryNode{}, false
	}
	if c.scanner == nil {
		return cloneNode(*c.manual), true
	}
	out := cloneNode(*c.scanner)
	if c.manual != nil {
		if out.Services == nil {
			out.Services = make(map[noderec.ServiceKey]noderec.ServiceStatus)
		}
		for k, v := range c.manual.Services {
			if _, ok := out.Services[k]; !ok {
				out.Services[k] = v
			}
		}
		out.Models = appendUnique(out.Models, c.manual.Models)
		if c.manual.ModelsByEngine != nil {
			if out.ModelsByEngine == nil {
				out.ModelsByEngine = make(map[string][]string)
			}
			for engine, models := range c.manual.ModelsByEngine {
				if _, exists := out.ModelsByEngine[engine]; !exists {
					out.ModelsByEngine[engine] = append([]string(nil), models...)
				}
			}
		}
		seen := make(map[string]bool)
		addresses := make([]string, 0, len(c.manual.CandidateIPs())+len(out.CandidateIPs()))
		for _, addr := range append(c.manual.CandidateIPs(), out.CandidateIPs()...) {
			if addr != "" && !seen[addr] {
				seen[addr] = true
				addresses = append(addresses, addr)
			}
		}
		if len(addresses) > 0 {
			out.IP = addresses[0]
		}
		if len(addresses) > 1 {
			out.IPs = addresses
		} else {
			out.IPs = nil
		}
	}
	return out, true
}

// Apply folds a scanner-owned claim into the directory.
func (d *Directory) Apply(method string, n noderec.DirectoryNode) { d.apply(false, method, n) }

// ApplyManual folds an authenticated manual/direct claim into the directory.
func (d *Directory) ApplyManual(method string, n noderec.DirectoryNode) { d.apply(true, method, n) }
func (d *Directory) apply(manual bool, method string, n noderec.DirectoryNode) {
	if n.HostUUID == "" {
		return
	}
	d.mu.Lock()
	c := d.nodes[n.HostUUID]
	before, bok := project(c)
	if method == noderec.NotifyNodeRemoved {
		if manual {
			c.manual = nil
		} else {
			c.scanner = nil
		}
	} else {
		cp := cloneNode(n)
		if manual {
			c.manual = &cp
		} else {
			c.scanner = &cp
		}
	}
	after, aok := project(c)
	if !aok {
		delete(d.nodes, n.HostUUID)
	} else {
		d.nodes[n.HostUUID] = c
	}
	changed := bok != aok || !reflect.DeepEqual(before, after)
	subs := make([]*Subscriber, 0, len(d.subs))
	if changed {
		for _, s := range d.subs {
			subs = append(subs, s)
		}
	}
	d.mu.Unlock()
	for _, s := range subs {
		d.Deliver(s)
	}
}

// Snapshot returns the current projected directory, optionally filtered by service.
func (d *Directory) Snapshot(filter noderec.ServiceKey) []noderec.DirectoryNode {
	d.mu.Lock()
	out := make([]noderec.DirectoryNode, 0, len(d.nodes))
	for _, c := range d.nodes {
		n, ok := project(c)
		if ok && (filter == "" || n.HasService(filter)) {
			out = append(out, n)
		}
	}
	d.mu.Unlock()
	sort.Slice(out, func(i, j int) bool { return out[i].HostUUID < out[j].HostUUID })
	return out
}
