// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/noderec"
)

func dcLeaf(t *testing.T, uuid string) (tls.Certificate, []byte, []byte) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	san, _ := url.Parse("urn:nvpair:node:" + uuid)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(time.Now().UnixNano()),
		Subject:      pkix.Name{CommonName: uuid},
		URIs:         []*url.URL{san},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, pub, priv)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, _ := x509.MarshalPKCS8PrivateKey(priv)
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER})
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	return cert, certPEM, keyPEM
}

func dcIdentity(t *testing.T, dir string, certPEM, keyPEM []byte) {
	t.Helper()
	os.WriteFile(filepath.Join(dir, "node.crt"), certPEM, 0600)
	os.WriteFile(filepath.Join(dir, "node.key"), keyPEM, 0600)
	body, _ := json.Marshal(map[string]any{"clusterId": "cluster-a", "epoch": 1})
	os.WriteFile(filepath.Join(dir, "admission.json"), body, 0600)
}

func dcPin(t *testing.T, dir, uuid string, certPEM []byte) {
	t.Helper()
	td := filepath.Join(dir, "trusted")
	os.MkdirAll(td, 0700)
	body, _ := json.Marshal(map[string]string{"nodeUuid": uuid, "certPem": string(certPEM)})
	os.WriteFile(filepath.Join(td, uuid+".json"), body, 0600)
}

// serveRoster stands up a roster endpoint that REQUIRES and verifies a client
// certificate, so the tests exercise mutual authentication rather than a
// server-only handshake. A request without the expected client principal is
// refused, which is what makes "no fallback after failure" meaningful.
func serveRoster(t *testing.T, cert tls.Certificate, expectClientDERPEM []byte, body string) (string, int, *int) {
	t.Helper()
	clientSeen := new(int)
	ln, err := tls.Listen("tcp", "127.0.0.1:0", &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAnyClientCert,
		MinVersion:   tls.VersionTLS12,
	})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc(rosterPath, func(w http.ResponseWriter, r *http.Request) {
		if r.TLS == nil || len(r.TLS.PeerCertificates) == 0 {
			http.Error(w, "client certificate required", http.StatusForbidden)
			return
		}
		// Actual certificate verification: exact DER equality against the
		// expected client leaf. A Common Name is a claim, not proof.
		if len(expectClientDERPEM) > 0 {
			block, _ := pem.Decode(expectClientDERPEM)
			if block == nil || !bytes.Equal(block.Bytes, r.TLS.PeerCertificates[0].Raw) {
				http.Error(w, "client certificate is not the expected leaf", http.StatusForbidden)
				return
			}
		}
		*clientSeen++
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(body))
	})
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })
	host, p, _ := net.SplitHostPort(ln.Addr().String())
	port, _ := strconv.Atoi(p)
	return host, port, clientSeen
}

func dcInfo(uuid string, port int) NodeInfoResponse {
	principal := uuid
	return NodeInfoResponse{
		HostUUID:    uuid,
		ClusterUUID: &principal,
		Services:    noderec.ServiceMap{noderec.ServiceCluster: port},
	}
}

func deadPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	_, p, _ := net.SplitHostPort(ln.Addr().String())
	ln.Close()
	port, _ := strconv.Atoi(p)
	return port
}

var selfPEM []byte

func dcManager(t *testing.T) (*Manager, string) {
	t.Helper()
	dir := t.TempDir()
	_, certPEM, keyPEM := dcLeaf(t, "self")
	dcIdentity(t, dir, certPEM, keyPEM)
	selfPEM = certPEM
	return &Manager{mesh: clustertrust.Open(dir)}, dir
}

func TestDirectConnectAcceptsExactPinnedDescriptorOverMutualTLS(t *testing.T) {
	m, dir := dcManager(t)
	certA, pemA, _ := dcLeaf(t, "host-a")
	dcPin(t, dir, "host-a", pemA)
	host, port, seen := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","services":{"ll":8081,"em":14319,"ec":14322}}}`)
	services, outcome := m.authenticateDirectConnect(host, dcInfo("host-a", port), "")
	if outcome != directConnectAccepted {
		t.Fatalf("outcome=%s", outcome)
	}
	if *seen == 0 {
		t.Fatal("server never verified a client certificate")
	}
	if services[noderec.ServiceLlamaCpp] != 8081 || services[noderec.ServiceEngineManager] != 14319 || services[noderec.ServiceEngineControl] != 14322 {
		t.Fatalf("services=%#v", services)
	}
}

// The authenticated descriptor governs every service; the plaintext answer is
// only a connection-port hint for the cluster-manager itself.
func TestDirectConnectAuthenticatedPortsOverridePlaintextDisagreement(t *testing.T) {
	m, dir := dcManager(t)
	certA, pemA, _ := dcLeaf(t, "host-a")
	dcPin(t, dir, "host-a", pemA)
	host, port, _ := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","services":{"ll":18435,"ol":11434}}}`)
	principal := "host-a"
	lying := NodeInfoResponse{
		HostUUID:    "host-a",
		ClusterUUID: &principal,
		Services: noderec.ServiceMap{
			noderec.ServiceCluster:  port,
			noderec.ServiceLlamaCpp: 8081,
			noderec.ServiceOllama:   1,
		},
	}
	services, outcome := m.authenticateDirectConnect(host, lying, "")
	if outcome != directConnectAccepted {
		t.Fatalf("outcome=%s", outcome)
	}
	if services[noderec.ServiceLlamaCpp] != 18435 || services[noderec.ServiceOllama] != 11434 {
		t.Fatalf("plaintext ports leaked into authenticated services: %#v", services)
	}
}

func TestDirectConnectOldPeerYieldsDescriptorAbsent(t *testing.T) {
	m, dir := dcManager(t)
	certA, pemA, _ := dcLeaf(t, "host-a")
	dcPin(t, dir, "host-a", pemA)
	host, port, _ := serveRoster(t, certA, selfPEM, `{"clusterId":"cluster-a"}`)
	services, outcome := m.authenticateDirectConnect(host, dcInfo("host-a", port), "")
	if outcome != directConnectAbsent || len(services) != 0 {
		t.Fatalf("outcome=%s services=%#v", outcome, services)
	}

	// Never-annotated and previously-annotated peers are covered by
	// TestDirectConnectStandaloneVersusDowngrade.
}

func TestDirectConnectFailuresYieldNoServicesAndNoFallback(t *testing.T) {
	m, dir := dcManager(t)
	certA, pemA, _ := dcLeaf(t, "host-a")
	certB, pemB, _ := dcLeaf(t, "host-b")
	certC, _, _ := dcLeaf(t, "host-c")
	dcPin(t, dir, "host-a", pemA)
	dcPin(t, dir, "host-b", pemB)

	valid := `{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","services":{"ll":18435}}}`
	siblingHost, siblingPort, _ := serveRoster(t, certB, nil, valid)
	unpinnedHost, unpinnedPort, _ := serveRoster(t, certC, nil,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-c","principal":"host-c","services":{"ll":18435}}}`)
	wrongPrincipalHost, wrongPrincipalPort, _ := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-b","services":{"ll":18435}}}`)
	wrongHostHost, wrongHostPort, _ := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-b","principal":"host-a","services":{"ll":18435}}}`)
	malformedHost, malformedPort, _ := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","host":"10.0.0.9","services":{"ll":18435}}}`)
	badPortHost, badPortPort, _ := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","services":{"ll":0}}}`)
	refusingHost, refusingPort, _ := serveRoster(t, certA, pemA, valid)
	okHost, _, _ := serveRoster(t, certA, selfPEM, valid)

	cases := []struct {
		name string
		host string
		port int
		uuid string
	}{
		{"another pinned member of the same cluster", siblingHost, siblingPort, "host-a"},
		{"unpinned principal", unpinnedHost, unpinnedPort, "host-c"},
		{"descriptor principal mismatch", wrongPrincipalHost, wrongPrincipalPort, "host-a"},
		{"descriptor hostUuid mismatch", wrongHostHost, wrongHostPort, "host-a"},
		{"malformed descriptor carrying a host field", malformedHost, malformedPort, "host-a"},
		{"invalid service port", badPortHost, badPortPort, "host-a"},
		{"server refuses our client principal", refusingHost, refusingPort, "host-a"},
		{"tls dial failure", okHost, deadPort(t), "host-a"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			services, outcome := m.authenticateDirectConnect(tc.host, dcInfo(tc.uuid, tc.port), "")
			if outcome != directConnectFailed {
				t.Fatalf("outcome=%s, want authentication_failed", outcome)
			}
			if len(services) != 0 {
				t.Fatalf("failed authentication returned services: %#v", services)
			}
		})
	}
}

func TestDirectConnectHintMismatchIsAuthenticationFailure(t *testing.T) {
	m, dir := dcManager(t)
	certA, pemA, _ := dcLeaf(t, "host-a")
	dcPin(t, dir, "host-a", pemA)
	host, port, _ := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","services":{"ll":8081}}}`)
	other := "cluster-a"
	mismatched := NodeInfoResponse{
		HostUUID:    "host-a",
		ClusterUUID: &other,
		Services:    noderec.ServiceMap{noderec.ServiceCluster: port},
	}
	if _, outcome := m.authenticateDirectConnect(host, mismatched, ""); outcome != directConnectFailed {
		t.Fatalf("outcome=%s, want authentication_failed", outcome)
	}
}

func TestDirectConnectAbsentRequiresAuthenticatedRoster(t *testing.T) {
	m, dir := dcManager(t)
	certA, pemA, _ := dcLeaf(t, "host-a")
	dcPin(t, dir, "host-a", pemA)
	host, port, _ := serveRoster(t, certA, selfPEM, `{"clusterId":"cluster-a"}`)

	// Only an authenticated roster that genuinely omits the descriptor is absent.
	if services, outcome := m.authenticateDirectConnect(host, dcInfo("host-a", port), ""); outcome != directConnectAbsent || len(services) != 0 {
		t.Fatalf("authenticated roster without descriptor: outcome=%s services=%#v", outcome, services)
	}

	// A stripped or missing plaintext clusterUuid must fail closed, never fall back.
}

func TestDirectConnectUnclusteredFailsClosed(t *testing.T) {
	// No identity and no pins: nothing can be authenticated, so no legacy fallback.
	bare := &Manager{mesh: clustertrust.Open(t.TempDir())}
	certA, _, _ := dcLeaf(t, "host-a")
	host, port, _ := serveRoster(t, certA, nil,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","services":{"ll":8081}}}`)
	services, outcome := bare.authenticateDirectConnect(host, dcInfo("host-a", port), "")
	if outcome != directConnectFailed || len(services) != 0 {
		t.Fatalf("unclustered outcome=%s services=%#v", outcome, services)
	}
}

// Case 1 versus case 4: an unannotated node that has never claimed a cluster is
// a standalone host and keeps legacy behavior; the same response from a node
// that previously claimed one is a downgrade and fails closed.
func TestDirectConnectStandaloneVersusDowngrade(t *testing.T) {
	m, dir := dcManager(t)
	certA, pemA, _ := dcLeaf(t, "host-a")
	dcPin(t, dir, "host-a", pemA)
	host, port, _ := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","services":{"ll":8081}}}`)

	blank := ""
	for _, tc := range []struct {
		name  string
		info  NodeInfoResponse
		prior string
		want  directConnectOutcome
	}{
		{"never annotated, absent field", NodeInfoResponse{HostUUID: "host-a", Services: noderec.ServiceMap{noderec.ServiceCluster: port}}, "", directConnectAbsent},
		{"never annotated, empty field", NodeInfoResponse{HostUUID: "host-a", ClusterUUID: &blank, Services: noderec.ServiceMap{noderec.ServiceCluster: port}}, "", directConnectAbsent},
		{"previously annotated, absent field", NodeInfoResponse{HostUUID: "host-a", Services: noderec.ServiceMap{noderec.ServiceCluster: port}}, "host-a", directConnectFailed},
		{"previously annotated, empty field", NodeInfoResponse{HostUUID: "host-a", ClusterUUID: &blank, Services: noderec.ServiceMap{noderec.ServiceCluster: port}}, "host-a", directConnectFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			services, outcome := m.authenticateDirectConnect(host, tc.info, tc.prior)
			if outcome != tc.want {
				t.Fatalf("outcome=%s want=%s", outcome, tc.want)
			}
			if len(services) != 0 {
				t.Fatalf("unauthenticated services surfaced: %#v", services)
			}
		})
	}
}

func TestClusterMemoryTracksAssertedIdentity(t *testing.T) {
	mem := &clusterMemory{seen: make(map[string]string)}
	if mem.prior("lab") != "" {
		t.Fatal("unseen node reported a prior identity")
	}
	mem.remember("lab", "")
	if mem.prior("lab") != "" {
		t.Fatal("empty identity was remembered")
	}
	mem.remember("lab", "cluster-a")
	if mem.prior("lab") != "cluster-a" {
		t.Fatal("asserted identity not remembered")
	}
	mem.forget("lab")
	if mem.prior("lab") != "" {
		t.Fatal("identity survived forget")
	}
}

// Deleting a manual entry is the ONE event that clears its downgrade memory.
// A probe failure, an authentication failure, or a vanished claim must not,
// because those are exactly the conditions a downgrade would manufacture.
func TestClusterMemoryLifecycleOnEntryDeletion(t *testing.T) {
	m, dir := dcManager(t)
	certA, pemA, _ := dcLeaf(t, "host-a")
	dcPin(t, dir, "host-a", pemA)
	host, port, _ := serveRoster(t, certA, selfPEM,
		`{"clusterId":"cluster-a","directConnect":{"hostUuid":"host-a","principal":"host-a","services":{"ll":8081}}}`)

	const id = "lab"
	t.Cleanup(func() { directConnectMemory.forget(id) })

	// 1. The node reports a cluster identity, which is remembered.
	directConnectMemory.remember(id, "host-a")
	if directConnectMemory.prior(id) != "host-a" {
		t.Fatal("clustered identity was not remembered")
	}

	// 2. While remembered, a stripped annotation is a downgrade and fails closed.
	stripped := NodeInfoResponse{HostUUID: "host-a", Services: noderec.ServiceMap{noderec.ServiceCluster: port}}
	if _, outcome := m.authenticateDirectConnect(host, stripped, directConnectMemory.prior(id)); outcome != directConnectFailed {
		t.Fatalf("stripped annotation while remembered: outcome=%s", outcome)
	}

	// 3. Authoritative entry deletion removes the observation.
	directConnectMemory.forget(id)
	if directConnectMemory.prior(id) != "" {
		t.Fatal("entry deletion did not clear the downgrade memory")
	}

	// 4. A deliberately re-added standalone entry is classified as case 1.
	services, outcome := m.authenticateDirectConnect(host, stripped, directConnectMemory.prior(id))
	if outcome != directConnectAbsent {
		t.Fatalf("re-added standalone entry: outcome=%s, want descriptor_absent", outcome)
	}
	if len(services) != 0 {
		t.Fatalf("standalone entry produced authenticated services: %#v", services)
	}
}
