// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
    JsonRpcSubprocess,
    type JsonRpcNotification
} from '@/electron/service-bridge/json-rpc-subprocess'

const mocks = vi.hoisted(() => ({
    bridgeState: {
        handleNotification: vi.fn(),
        applyEngineManagerStatus: vi.fn(),
        getSelfId: vi.fn<() => string | null>(() => null),
        getProxyPort: vi.fn<() => number | null>(() => null),
        setSelfId: vi.fn<(id: string) => void>()
    },
    emitBridgePush: vi.fn()
}))

vi.mock('electron', () => ({
    app: {
        isPackaged: false,
        getAppPath: () => process.cwd()
    }
}))

vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

vi.mock('@/electron/config/ui-config', () => ({
    isFirstRun: () => false
}))

vi.mock('@/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: mocks.emitBridgePush
}))

vi.mock('@/electron/service-bridge/manual-nodes-store', () => ({
    listManualNodeEntries: () => []
}))

vi.mock('@/electron/service-bridge/node-info-poller', () => ({
    startNodeInfoPoller: vi.fn(),
    stopNodeInfoPoller: vi.fn()
}))

vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => mocks.bridgeState,
    isUpstreamUnreachableError: () => false,
    parseServiceErrors: () => [],
    parseWorkloadsInitial: () => [],
    PROXY_ENGINES: ['ollama', 'lm-studio', 'llamacpp']
}))

import {
    getModularSupervisor,
    ModularStartupTimeoutError
} from '@/electron/service-bridge/modular-supervisor'

interface ReadinessHarness {
    readonly ready: boolean
    processes: Map<string, JsonRpcSubprocess>
    isReady: boolean
    brokerReady: boolean
    brokerHydrationDone: boolean
    readinessReported: boolean
    nextReadinessWaiterId: number
    readinessWaiters: Map<number, { timeout: ReturnType<typeof setTimeout> }>
    onReady?: () => void
    onBrokerReady: () => Promise<void>
    setOnReady: (callback: () => void) => void
    waitUntilReady: (timeoutMs: number) => Promise<void>
    handleNotification: (notification: JsonRpcNotification) => void
    attachChildHandlers: (child: JsonRpcSubprocess) => void
}

const supervisor = getModularSupervisor() as unknown as ReadinessHarness

function notify(method: string, params?: JsonRpcNotification['params']): void {
    supervisor.handleNotification({ source: 'broker', method, params })
}

describe('modular supervisor readiness', () => {
    beforeEach(() => {
        vi.useFakeTimers()
        for (const waiter of supervisor.readinessWaiters.values()) {
            clearTimeout(waiter.timeout)
        }
        supervisor.readinessWaiters.clear()
        supervisor.nextReadinessWaiterId = 0
        supervisor.readinessReported = false
        supervisor.isReady = true
        supervisor.brokerReady = false
        supervisor.brokerHydrationDone = false
        supervisor.processes.clear()

        const localBridges = Reflect.get(supervisor, 'localBridges')
        if (!(localBridges instanceof Map)) {
            throw new Error('localBridges is unavailable')
        }
        localBridges.clear()

        supervisor.onReady = undefined
        supervisor.onBrokerReady = vi.fn().mockResolvedValue(undefined)
    })

    afterEach(() => {
        for (const waiter of supervisor.readinessWaiters.values()) {
            clearTimeout(waiter.timeout)
        }
        supervisor.readinessWaiters.clear()
        vi.useRealTimers()
    })

    it('requires app:ready but not optional proxy readiness', async () => {
        const onReady = vi.fn()
        supervisor.setOnReady(onReady)
        const waiting = supervisor.waitUntilReady(1_000)
        let settled = false
        void waiting.then(() => {
            settled = true
        })

        notify('proxy:ready', { port: 11434 })
        await Promise.resolve()

        expect(supervisor.ready).toBe(false)
        expect(settled).toBe(false)
        expect(onReady).not.toHaveBeenCalled()

        notify('app:ready')

        await expect(waiting).resolves.toBeUndefined()
        expect(supervisor.ready).toBe(true)
        expect(onReady).toHaveBeenCalledOnce()
    })

    it('keeps the service ready and refreshes capability state when the proxy arrives late', async () => {
        const onReady = vi.fn()
        supervisor.setOnReady(onReady)

        notify('app:ready')
        await Promise.resolve()
        mocks.emitBridgePush.mockClear()
        supervisor.brokerHydrationDone = true

        notify('proxy:error', { message: 'address already in use' })
        notify('proxy:ready', { port: 11435 })

        expect(supervisor.ready).toBe(true)
        expect(onReady).toHaveBeenCalledOnce()
        expect(mocks.bridgeState.handleNotification).toHaveBeenLastCalledWith({
            source: 'proxy',
            method: 'ready',
            params: { port: 11435 }
        })
        expect(mocks.emitBridgePush).toHaveBeenCalledWith('state:request-refresh', undefined)
    })

    it('diagnoses a missing app:ready and still recovers from a late notification', async () => {
        const onReady = vi.fn()
        supervisor.setOnReady(onReady)
        const waiting = supervisor.waitUntilReady(15_000)
        const rejection = expect(waiting).rejects.toEqual(new ModularStartupTimeoutError(15_000))

        await vi.advanceTimersByTimeAsync(15_000)
        await rejection

        notify('app:ready')

        expect(supervisor.ready).toBe(true)
        expect(onReady).toHaveBeenCalledOnce()
    })

    it('re-bridges a running local engine when self identity resolves after startup', async () => {
        const resolvedId = '7a6c3d78-58f7-4b1f-a88a-5e3c8fdc9f19'
        let selfId: string | null = null
        let proxyPort: number | null = null
        let settleFirstRelay = () => {}

        const firstRelay = new Promise<void>(resolve => {
            settleFirstRelay = resolve
        })
        const callProxy = vi.fn(() => firstRelay)

        mocks.bridgeState.getSelfId.mockImplementation(() => selfId)
        mocks.bridgeState.getProxyPort.mockImplementation(() => proxyPort)
        mocks.bridgeState.setSelfId.mockImplementation((id: string) => {
            selfId = id
        })

        Reflect.set(
            supervisor,
            'callProcess',
            vi.fn(async (_name: string, method: string) => {
                if (method === 'cluster:get-node-id') {
                    return { nodeUuid: resolvedId }
                }
                throw new Error(`unexpected broker method ${method}`)
            })
        )
        Reflect.set(supervisor, 'callProxy', callProxy)
        supervisor.processes.set('broker', new JsonRpcSubprocess('broker', 'test-broker'))

        const updateBridge = Reflect.get(supervisor, 'updateLocalNodeBridgeFromEngineState')
        if (typeof updateBridge !== 'function') {
            throw new Error('updateLocalNodeBridgeFromEngineState is unavailable')
        }

        // Engine state, proxy readiness, and broker readiness all arrive while
        // identity is absent. Each bridge trigger must remain dormant.
        Reflect.apply(updateBridge, supervisor, [
            { engine: 'llamacpp', running: true, port: 18434 }
        ])

        proxyPort = 18435
        supervisor.handleNotification({
            source: 'llamacpp-proxy',
            method: 'ready',
            params: { port: 18435 }
        })

        notify('app:ready')
        await Promise.resolve()

        expect(callProxy).not.toHaveBeenCalled()

        const resolveSelfId = Reflect.get(supervisor, 'resolveSelfId')
        if (typeof resolveSelfId !== 'function') {
            throw new Error('resolveSelfId is unavailable')
        }

        await Reflect.apply(resolveSelfId, supervisor, [])

        await vi.waitFor(() => {
            expect(callProxy).toHaveBeenCalledTimes(1)
        })
        expect(callProxy).toHaveBeenCalledWith('llamacpp', 'node/add-manual', {
            id: resolvedId,
            host: '127.0.0.1',
            port: 18434,
            addresses: ['127.0.0.1']
        })

        // Production deliberately keeps reconciliation fire-and-forget. Resolve
        // the mocked RPC, then wait for reconcileLocalNodeBridge to record its
        // bridged ID and port before testing repeated identity resolution.
        settleFirstRelay()

        const localBridges = Reflect.get(supervisor, 'localBridges')
        if (!(localBridges instanceof Map)) {
            throw new Error('localBridges is unavailable')
        }

        await vi.waitFor(() => {
            const bridge = localBridges.get('llamacpp')
            if (typeof bridge !== 'object' || bridge === null) {
                throw new Error('llama.cpp local bridge is unavailable')
            }
            expect(Reflect.get(bridge, 'bridgedId')).toBe(resolvedId)
            expect(Reflect.get(bridge, 'bridgedPort')).toBe(18434)
        })

        await Reflect.apply(resolveSelfId, supervisor, [])
        await Promise.resolve()

        expect(callProxy).toHaveBeenCalledTimes(1)
    })

    it('bridges a hydrated running engine after self identity resolves first', async () => {
        const resolvedId = '8bb927c3-4f07-4ad7-9bb6-e2cf3f88676a'
        let selfId: string | null = null
        let proxyPort: number | null = null

        mocks.bridgeState.getSelfId.mockImplementation(() => selfId)
        mocks.bridgeState.getProxyPort.mockImplementation(() => proxyPort)
        mocks.bridgeState.setSelfId.mockImplementation((id: string) => {
            selfId = id
        })

        const callProcess = vi.fn(async (_name: string, method: string) => {
            if (method === 'cluster:get-node-id') {
                return { nodeUuid: resolvedId }
            }
            if (method === 'engine:get-installed') {
                return {
                    engines: [
                        {
                            engine: 'llamacpp',
                            displayName: 'llama.cpp',
                            installed: true,
                            running: true,
                            healthy: true,
                            port: 18434
                        }
                    ]
                }
            }
            if (method === 'engine:action') {
                return { models: [] }
            }
            throw new Error(`unexpected broker method ${method}`)
        })
        const callProxy = vi.fn().mockResolvedValue({ added: true })

        Reflect.set(supervisor, 'callProcess', callProcess)
        Reflect.set(supervisor, 'callProxy', callProxy)
        supervisor.processes.set('broker', new JsonRpcSubprocess('broker', 'test-broker'))

        const resolveSelfId = Reflect.get(supervisor, 'resolveSelfId')
        if (typeof resolveSelfId !== 'function') {
            throw new Error('resolveSelfId is unavailable')
        }

        const hydrateEngineManager = Reflect.get(supervisor, 'hydrateEngineManager')
        if (typeof hydrateEngineManager !== 'function') {
            throw new Error('hydrateEngineManager is unavailable')
        }

        // Production startup resolves identity before engine hydration.
        await Reflect.apply(resolveSelfId, supervisor, [])
        expect(selfId).toBe(resolvedId)
        expect(callProxy).not.toHaveBeenCalled()

        // Proxy readiness can also precede hydration. No bridge is possible yet
        // because the desired running state and backend port are unknown.
        proxyPort = 18435
        supervisor.handleNotification({
            source: 'llamacpp-proxy',
            method: 'ready',
            params: { port: 18435 }
        })
        await Promise.resolve()
        expect(callProxy).not.toHaveBeenCalled()

        // Hydration supplies running=true and port=18434 and must itself trigger
        // the bridge without waiting for engine:state-changed.
        await Reflect.apply(hydrateEngineManager, supervisor, [])

        await vi.waitFor(() => {
            expect(callProxy).toHaveBeenCalledTimes(1)
        })
        expect(callProxy).toHaveBeenCalledWith('llamacpp', 'node/add-manual', {
            id: resolvedId,
            host: '127.0.0.1',
            port: 18434,
            addresses: ['127.0.0.1']
        })

        const localBridges = Reflect.get(supervisor, 'localBridges')
        if (!(localBridges instanceof Map)) {
            throw new Error('localBridges is unavailable')
        }

        await vi.waitFor(() => {
            const bridge = localBridges.get('llamacpp')
            if (typeof bridge !== 'object' || bridge === null) {
                throw new Error('llama.cpp local bridge is unavailable')
            }
            expect(Reflect.get(bridge, 'bridgedId')).toBe(resolvedId)
            expect(Reflect.get(bridge, 'bridgedPort')).toBe(18434)
        })

        // Repeating authoritative hydration remains idempotent.
        await Reflect.apply(hydrateEngineManager, supervisor, [])
        await Promise.resolve()

        expect(callProxy).toHaveBeenCalledTimes(1)
    })

    it('ignores readiness from a replaced broker generation', () => {
        const oldBroker = new JsonRpcSubprocess('broker', 'old-broker')
        const newBroker = new JsonRpcSubprocess('broker', 'new-broker')
        supervisor.processes.set('broker', oldBroker)
        supervisor.attachChildHandlers(oldBroker)

        oldBroker.emit('exit', { source: 'broker', code: 1 })
        supervisor.processes.set('broker', newBroker)
        supervisor.attachChildHandlers(newBroker)
        supervisor.isReady = true

        oldBroker.emit('notification', { source: 'broker', method: 'app:ready' })

        expect(supervisor.ready).toBe(false)

        newBroker.emit('notification', { source: 'broker', method: 'app:ready' })

        expect(supervisor.ready).toBe(true)

        oldBroker.emit('exit', { source: 'broker', code: 1 })

        expect(supervisor.ready).toBe(true)
    })
})
