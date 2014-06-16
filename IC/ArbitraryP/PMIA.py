''' Implementation of PMIA algorithm [1].

[1] -- Scalable Influence Maximization for Prevalent Viral Marketing in Large-Scale Social Networks.
'''

def computeAP(ap, v, S, PMIIA, Ep):
    ''' Assumption: PMIIA is a directed tree.
    '''
    # Find leaves
    us = []
    for node in PMIIA:
        if not PMIIA.in_edges([node]):
            us.append(node)
    # Find ap moving from leaves to the root
    for u in us:
        if u in S:
            ap[(u, v)] = 1
        elif not PMIIA.in_edges([u]):
            ap[(u, v)] = 0
        else:
            in_edges = PMIIA.in_edges([u], data=True)
            prod = 1
            # TODO check if w,_ == w,u
            for w, _, edata in in_edges:
                prod *= 1 - ap[(w, v)]*(1 - (1 - Ep[(w, u)])**edata["weight"])
            ap[(u, v)] = 1 - prod

        out_edges = PMIIA.out_edges([u])
        #TODO check if u, out_node == out_edges
        us.extend([out_node for _, out_node in out_edges])

def PMIA(G, k, theta, Ep):
    # initialization
    S = []
    IncInf = dict(zip(G.nodes(), [0]*len(G)))
    PMIIA = dict()
    PMIOA = dict()
    ap = dict()
    alpha = dict()
    for v in G:
        PMIIA[v] = computePMIIA(v, theta, S)
        updateAlpha(alpha, v, S, PMIIA[v])
        for u in PMIIA[v]:
            ap[(u, v)] = 0 # ap of u node in PMIIA[v]
            IncInf[u] += alpha[(v,u)]*(1 - ap[(u, v)])

    # main loop
    for i in range(k):
        u, _ = max(IncInf.iteritems(), key = lambda (dk, dv): dv)
        IncInf.pop(u) # exclude node u for next iterations
        PMIOA[u] = computePMIOA(u, theta, S)
        for v in PMIOA[u]:
            for w in PMIIA[v]:
                if w not in S:
                    IncInf[w] -= alpha[(v,w)]*(1 - ap[(w, PMIIA[v])])

        S.append(u)

        for v in PMIOA[u]:
            if v != u:
                PMIIA[v] = computePMIIA(v, theta, S)
                updateAP(ap, v, S, PMIIA[v], Ep)
                updateAlphas(alpha, v, S, PMIIA[v])
                # add new incremental influence
                for w in PMIIA[v]:
                    if w not in S:
                        IncInf[w] += alpha[(v, w)]*(1 - ap[(w, v)])

    return S