__author__ = 'ivanovsergey'

class Vertex(object):
    def __init__(self,key):
        self.id = key
        self.connectedTo = dict()

    def addNeighbor(self,v,weight=0):
        self.connectedTo[v] = self.connectedTo.get(v, 0) + weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, key):
        for v in self.connectedTo.keys():
            if v.id == key:
                return self.connectedTo[v]

class Graph(object):
    def __init__(self):
        self.vertDict = dict()
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices += 1
        newVertex = Vertex(key)
        self.vertDict[key] = newVertex
        return newVertex

    def getVertex(self,key):
        if key in self.vertDict:
            return self.vertDict[key]

    def __contains__(self,n):
        return n in self.vertDict

    def addEdge(self,key1,key2,cost=1, d = False):
        ''' d -- graph is directional
        '''
        if key1 not in self.vertDict:
            nv = self.addVertex(key1)
        if key2 not in self.vertDict:
            nv = self.addVertex(key2)
        self.vertDict[key1].addNeighbor(self.vertDict[key2], cost)
        if not d:
            self.vertDict[key2].addNeighbor(self.vertDict[key1], cost)

    def getVertices(self):
        return self.vertDict.keys()

    def __iter__(self):
        return iter(self.vertDict.values())

    def __len__(self):
        return self.numVertices

    def excludeVertex(self, key):
        u = self.vertDict[key]
        neighbors = u.getConnections()
        for v in neighbors:
            v.connectedTo.pop(u)
        self.vertDict.pop(key)
        self.numVertices -= 1
        