import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from bisect import bisect_left
from functools import cmp_to_key
from PolynomialFit import *

EDIT_ASSERTS = True

##########################################################
#              Partial Order Functions                   #
##########################################################

def TotalOrder2DX(N1, N2):
    """
    For trees embedded in the plane corresponding to
    1D functions, make a total order based on the X
    coordinate
    """
    if N1.X[0] <= N2.X[0]:
        return -1
    return 1

def PartialOrder3DXY(N1, N2):
    """
    For trees build on functions over R2 (such as SSMs)
    #which live in 3D, use the domain to create a partial
    order
    """
    [X1, Y1] = [N1.X[0], N1.X[1]]
    [X2, Y2] = [N2.X[0], N2.X[1]]
    if X1 <= X2 and Y1 <= Y2:
        return -1
    elif X1 >= X2 and Y1 >= Y2:
        return 1
    return 0


##########################################################
#           Merge Tree Utility Functions                 #
##########################################################

def isAncestor(node, ancestor):
    if not node.parent:
        return False
    if node.parent == ancestor:
        return True
    return isAncestor(node.parent, ancestor)


def getParentNotSubdivided(N):
    if not N.parent:
        return None
    if N.parent.subdivided:
        return getParentNotSubdivided(N.parent)
    else:
        return N.parent

def subdivideTreesMutual(TA, TB, includeLeaves = False):
    valsA = TA.getfValsSorted(includeLeaves)
    valsB = TB.getfValsSorted(includeLeaves)
    #Subdivide both edges to make sure internal nodes get matched to internal nodes by horizontal lines
    #vals = np.array(valsA.tolist() + valsB.tolist())
    #vals = np.sort(np.unique(vals))
    TB.subdivideFromValues(valsA)
    TA.subdivideFromValues(valsB)
    TA.updateNodesList()
    TB.updateNodesList()

##########################################################
#                   Merge Tree Maps                      #
##########################################################

class ChiralMap(object):
    def __init__(self, TA, TB):
        self.TA = TA
        self.TB = TB
        self.cost = np.inf
        self.Map = {}
        self.mapsChecked = 0
        self.BsNotHit = []

def drawMap(ChiralMap, offsetA, offsetB, yres = 0.5, drawSubdivided = True, drawCurved = True):
    (TA, TB) = (ChiralMap.TA, ChiralMap.TB)
    #First draw the two trees
    ax = plt.subplot(111)
    TA.render(offsetA, drawCurved = drawCurved, drawSubdivided = drawSubdivided)
    TB.render(offsetB, drawCurved = drawCurved, drawSubdivided = drawSubdivided)
    #Put y ticks at every unique y value
    yvals = TA.getfValsSorted().tolist() + TB.getfValsSorted().tolist()
    yvals = np.sort(np.unique(np.array(yvals)))
    ax.set_yticks(yvals)
    plt.grid()
    plt.title("Cost = %g\nmapsChecked = %s"%(ChiralMap.cost, ChiralMap.mapsChecked))

    #Now draw arcs between matched nodes and draw Xs over
    #nodes that didn't get matched
    Map = ChiralMap.Map
    for A in Map:
        B = Map[A]
        ax = A.X + offsetA
        if not A.subdivided or drawSubdivided:
            if not B:
                #Draw an X over this node
                plt.scatter([ax[0]], [ax[1]], 300, 'k', 'x')
            else:
                bx = B.X + offsetB# + 0.1*np.random.randn(2)
                plt.plot([ax[0], bx[0]], [ax[1], bx[1]], 'b')
    for B in ChiralMap.BsNotHit:
        if not B.subdivided or drawSubdivided:
            bx = B.X + offsetB
            plt.scatter([bx[0]], [bx[1]], 300, 'k', 'x')

class DebugOffsets(object):
    def __init__(self, offsetA, offsetB):
        self.offsetA = offsetA
        self.offsetB = offsetB


##########################################################
#               Core Merge Tree Objects                  #
##########################################################

class MergeNode(object):
    """
    X: Rendering position
    fval: Function value (assumed to be last coordinate of X if not provided)
    subdivided: Whether this node was added in a subdivision
    """
    def __init__(self, X, fval = None, subdivided = False):
        self.parent = None
        self.children = []
        self.X = np.array(X, dtype=np.float64)
        if not fval:
            self.fval = X[-1]
        else:
            self.fval = fval
        self.subdivided = subdivided
        self.idx = -1

    def getfVal(self):
        return self.fval
    
    def setfVal(self, fval):
        self.fval = fval

    def addChild(self, N):
        self.children.append(N)
        N.parent = self

    def addChildren(self, arr):
        for C in arr:
            self.addChild(C)

    def cloneValOnly(self):
        ret = MergeNode(self.X)
        return ret

    def __str__(self):
        return "Node: X = %s, Subdivided = %i"%(self.X, self.subdivided)

class MergeTree(object):
    """
    Holds nodes starting at root, and a table of partial order info (
    -1 for less than, 1 for greater than, 0 for undefined)
    """
    def __init__(self, orderFn):
        self.root = None
        self.orderFn = orderFn
        self.fVals = []
        self.nodesList = []

    def clone(self):
        T = MergeTree(self.orderFn)
        T.root = self.root.cloneValOnly()
        stack = [(self.root, T.root)]
        while len(stack) > 0:
            (node, newnode) = stack.pop()
            for C in node.children:
                newC = C.cloneValOnly()
                newnode.addChild(newC)
                stack.append((C, newC))
        return T

    def addOffsetRec(self, node, offset):
        node.X += offset
        for C in node.children:
            self.addOffsetRec(C, offset)

    def addOffset(self, offset):
        self.addOffsetRec(self.root, offset)

    def getCriticalArrayRec(self, node, arr):
        """
        Do an inorder traversal of the tree to get the
        critical array
        """
        C = node.children
        L = None
        R = None
        if len(C) > 0:
            if C[0].X[0] < node.X[0]:
                L = C[0]
            else:
                R = C[0]
        if len(C) > 1:
            if C[1].X[0] < node.X[0]:
                L = C[1]
            else:
                R = C[1]
        if L:
            self.getCriticalArrayRec(L, arr)
        arr.append(node.getfVal())
        if R:
            self.getCriticalArrayRec(R, arr)


    def getCriticalArray(self):
        ret = []
        self.getCriticalArrayRec(self.root, ret)
        ret = np.array(ret)
        return ret

    def renderRec(self, node, offset, drawSubdivided = True, drawCurved = True, lineWidth = 3, pointSize = 200):
        X = node.X + offset
        if node.subdivided:
            #Render new nodes blue
            if drawSubdivided:
                plt.scatter(X[0], X[1], pointSize, 'r')
        else:
            plt.scatter(X[0], X[1], pointSize, 'k')
        if node.parent:
            Y = node.parent.X + offset
            if drawCurved:
                #Draw edge arc
                [x1, y1, x3, y3] = [X[0], X[1], Y[0], Y[1]]
                x2 = 0.5*x1 + 0.5*x3
                y2 = 0.25*y1 + 0.75*y3
                xs = np.linspace(x1, x3, 50)
                X = np.array([[x1, y1], [x2, y2], [x3, y3]])
                Y = polyFit(X, xs, doPlot = False)
                plt.plot(Y[:, 0], Y[:, 1], 'k', linewidth = lineWidth)
            else:
                plt.plot([X[0], Y[0]], [X[1], Y[1]], 'k', lineWidth = lineWidth)
        for C in node.children:
            self.renderRec(C, offset, drawSubdivided, drawCurved, lineWidth, pointSize)

    def render(self, offset, drawSubdivided = True, drawCurved = True, lineWidth = 3, pointSize = 200, ):
        self.renderRec(self.root, offset, drawSubdivided, drawCurved, lineWidth, pointSize)

    def sortChildrenTotalOrderRec(self, N):
        N.children = sorted(N.children, key=cmp_to_key(self.orderFn))
        for C in N.children:
            self.sortChildrenTotalOrderRec(C)


    def sortChildrenTotalOrder(self):
        """
        Sort the children by their total order (behavior undefined
        if orderFn is a partial order)
        """
        self.sortChildrenTotalOrderRec(self.root)

    def getfValsSortedRec(self, node, includeLeaves = True):
        if includeLeaves or len(node.children) > 0:
            self.fVals.append(node.getfVal())
        for n in node.children:
            self.getfValsSortedRec(n, includeLeaves)

    def getfValsSorted(self, includeLeaves = True):
        """Get a sorted list of all of the function values"""
        self.fVals = []
        self.getfValsSortedRec(self.root, includeLeaves)
        self.fVals = sorted(self.fVals)
        self.fVals = np.unique(self.fVals)
        return self.fVals

    def updateNodesListRec(self, N):
        N.idx = len(self.nodesList)
        self.nodesList.append(N)
        for C in N.children:
            self.updateNodesListRec(C)

    def updateNodesList(self):
        self.nodesList = []
        self.updateNodesListRec(self.root)
    
    def containsNodeRec(self, node, thisN):
        """
        Recursive helper function for node containment
        :param node: The node object being sought
        :param thisN: The current node in the tree search
        """
        if thisN == node:
            return True
        for C in thisN.children:
            if self.containsNodeRec(node, C):
                return True
        return False
    
    def containsNode(self, node):
        return self.containsNodeRec(node, self.root)

    def getCriticalPtsList(self):
        """Return an Nxd numpy array of the N critical points"""
        self.updateNodesList()
        N = len(self.nodesList)
        X = np.zeros((N, self.nodesList[0].X.size))
        for i in range(N):
            X[i, :] = self.nodesList[i].X
        return X

    def subdivideEdgesRec(self, N1, vals, hi):
        """
        Recursive helper function for subdividing edges with all elements
        in "vals"
        hi: Index such that vals[0:hi] < N1.fVal
        """
        b = N1.getfVal()
        if b <= vals[0]:
            return
        for i in range(len(N1.children)):
            N2 = N1.children[i]
            a = N2.getfVal()
            #Figure out the elements in vals which
            #are in the open interval (a, b)
            lo = bisect_left(vals, a)
            splitVals = []
            for k in range(lo, hi+1):
                if k >= len(vals):
                    continue
                if vals[k] <= a or vals[k] >= b:
                    continue
                splitVals.append(vals[k])
            #Nodes were sorted in increasing order of height
            #but need to add them in decreasing order
            #for the parent relationships to work out
            splitVals = splitVals[::-1]
            if len(splitVals) > 0:
                #Now split the edge between N1 and N2
                newNodes = []
                for k in range(len(splitVals)):
                    t = (splitVals[k] - a)/float(b - a)
                    X = t*N1.X + (1-t)*N2.X
                    N = MergeNode(X, subdivided = True)
                    if k > 0:
                        newNodes[k-1].addChild(N)
                    newNodes.append(N)
                #The last node is connected to N2
                newNodes[-1].addChild(N2)
                #Replace N2 in N1's children list with
                #the first node in newNodes
                N1.children[i] = newNodes[0]
                newNodes[0].parent = N1
            self.subdivideEdgesRec(N2, vals, lo)

    def subdivideFromValues(self, vals):
        """
        Note: For simplicity of implementation, this
        function assumes that parent nodes have higher
        function values than their children
        """
        hi = bisect_left(vals, self.root.getfVal())
        self.subdivideEdgesRec(self.root, vals, hi)

    def clearSubdividedNodes(self):
        """Remove all subdivided nodes"""
        print("TODO")
    
    ##########################################################
    #                     Edit Operations                    #
    ##########################################################
    def changeF(self, v, y, asserts = EDIT_ASSERTS):
        """
        Change a function value f(v) to y, with f(c) < y < f(p(v)) for all
        children c of v.
        :returns: |f(v) - y|
        """
        if asserts:
            #Make sure this node is in the node list
            if not self.containsNode(v):
                print("Error: Trying to change function value of node not in tree")
                return -1
            #Make sure the f edit value is below the parent and above
            #all children
            pval = np.inf
            if v.parent:
                pval = v.parent.getfVal()
            if y > pval:
                print("Error: Moving node to %g above its parent at %g"(y, pval))
                return -1
            for c in v.children:
                cval = c.getfVal()
                if y < cval:
                    print("Error: Moving node to %g below one of its children at %g"%(y, cval))
                    return -1
        yorig = v.getfVal()
        v.setfVal(y)
        v.X[-1] = y
        return {'cost':np.abs(y - yorig)}
    
    def collapseEdge(self, a, b, asserts = EDIT_ASSERTS):
        """
        Collapse an edge (a, b) in the tree, where neither a nor b is a regular vertex.
        The cost is f(a) - f(b).  Vertex b is deleted and all of the children of b are
        inherited by a
        :returns: Cost f(a) - f(b)
        """
        if asserts:
            if not self.containsNode(a):
                print("Error: Trying to collapse to vertex that doesn't exist in tree")
                return -1
            if not self.containsNode(b):
                print("Error: Trying to collapse a vertex that doesn't exist in tree")
                return -1
        if not b in a.children:
            print("Error: Trying to collapse edge (a, b), but b is not a child of a")
            return -1
        a.children.remove(b)
        a.children += b.children
        for c in b.children:
            c.parent = a
        aval = a.getfVal()
        bval = b.getfVal()
        if bval > aval:
            print("Error: bval %g > aval %g when collapsing edge (a, b)"%(bval, aval))
            return -1
        return {'cost':aval - bval}
    
    def splitChildren(self, a, fv, c1, c2, asserts = EDIT_ASSERTS):
        """
        Split the children c = c1 union c2 of node a by creating a new node v
        as a child of a, with f(a) > f(v) > f(c2), so that a now has the children
        c1,v and v has children c2.
        :returns: Cost f(a) - f(v)
        """
        if asserts:
            if not self.containsNode(a):
                print("Trying to split the children of a node that's not in the tree")
                return -1
            #Make sure a's children are c1 union c2
            if not set(c1 + c2) == set(a.children):
                print("Error: Trying to do a children split at a, but the union of c1 and c2 is not a's children")
                return -1
            #Make sure f(a) > f(v) > f(c) for all children c of a
            if fv > a.getfVal():
                print("Error: Trying to split children of a, but split vertex at %g is above a at %g"%(fv, a.getfVal()))
                return -1
            for c in c2:
                if fv < c.getfVal():
                    print("Error: Trying to split children of a, but split vertex at %g is below one of a's children at %g"%(fv, c.getfVal()))
                    return -1
        
        #Pick one of the children in c2 and use it to create an interpolated position
        X = np.array(a.X)
        if len(c2) > 0:
            c = c2[0]
            t = (fv - c.getfVal())/(a.getfVal() - c.getfVal())
            X = t*X + (1-t)*c.X
        v = MergeNode(X, fv)
        a.children = c1 + [v]
        v.parent = a
        v.children = c2
        for c in c2:
            c.parent = v
        return {'cost':a.getfVal() - fv, 'v':v}
    
    def addEdge(self, a, fb, asserts = EDIT_ASSERTS):
        """
        Add an edge (a, b) rooted at a regular node a with function value 
        fb = y <= f(a)
        :returns: cost f(a) - fb
        """
        if asserts:
            if not self.containsNode(a):
                print("Trying to add an edge to a vertex that's not in the tree")
                return -1
        if fb > a.getfVal():
            print("Error: adding leaf edge (a, b) at height %g above vertex a at %g"%(fb, a.getfVal()))
            return -1
        if not len(a.children) == 1:
            print("Error: Adding edge at node which is not regular (%i children)"%(len(a.children)))
            return -1
        X = np.array(a.X)
        X[-1] = fb
        b = MergeNode(X, fb)
        b.parent = a
        a.children.append(b)
        return {'cost':a.getfVal() - fb, 'v':b}
    
    def deleteRegVertex(self, v, asserts = EDIT_ASSERTS):
        """
        Delete a regular vertex at a cost of zero.  All of v's children
        become children of v's parent
        """
        if asserts:
            if not self.containsNode(v):
                print("Trying to delete a regular vertex that's not in the tree")
                return -1
        if not len(v.children) == 1 or not v.parent:
            print("Error: Trying to delete a vertex which is not a regular vertex")
            return -1
        v.parent.children.remove(v)
        v.parent.children += v.children
        for c in v.children:
            c.parent = v.parent
        return {'cost':0}
    
    def insertRegVertex(self, a, b, y, asserts = EDIT_ASSERTS):
        """
        Insert a regular vertex v at height y on the edge (a, b) with f(b) <= y <= f(a)
        at a cost of 0
        """
        if asserts:
            if not self.containsNode(a):
                print("Error: Trying to collapse to vertex that doesn't exist in tree")
                return -1
            if not self.containsNode(b):
                print("Error: Trying to collapse a vertex that doesn't exist in tree")
                return -1
            if y > a.getfVal():
                print("Error: Trying to insert an internal node at height %g above segment with max %g"%(y, a.getfVal()))
                return -1
            if y < b.getfVal():
                print("Error: Trying to insert an internal node at height %g below segment with min %g"%(y, b.getfVal()))
        
        if not b in a.children:
            print("Error: Trying to insert regular vertex into (a, b), but (a, b) is not an edge in the tree")
            return -1
        a.children.remove(b)
        
        t = (y - b.getfVal())/(a.getfVal() - b.getfVal())
        v = MergeNode(t*a.X + (1-t)*b.X, y)
        #E = E union (a, v)
        a.children.append(v)
        v.parent = a
        #E = E union (v, b)
        v.children = [b]
        b.parent = v
        return {'cost':0, 'v':v}

def UFFind(UFP, u):
    """
    Union find "find" with path-compression
    :param UFP: A list of pointers to reprsentative nodes
    :param u: Index of the node to find
    :return: Index of the representative of the component of u
    """
    if not (UFP[u] == u):
        UFP[u] = UFFind(UFP, UFP[u])
    return UFP[u]

def UFUnion(UFP, u, v, idxorder):
    """
    Union find "union" with early birth-based merging
    (similar to rank-based merging...not sure if exactly the
    same theoretical running time)
    """
    u = UFFind(UFP, u)
    v = UFFind(UFP, v)
    if u == v:
        return #Already in union
    [ufirst, usecond] = [u, v]
    if idxorder[v] < idxorder[u]:
        [ufirst, usecond] = [v, u]
    UFP[usecond] = ufirst

def wrapMergeTreeTimeSeries(MT, PS, X):
    """
    s is a time series from the GDA library, X is an Nx2 numpy
    array of the corresponding coordinates
    Return Merge Tree Object
    """
    #First extract merge tree
    T = MergeTree(TotalOrder2DX)
    y = X[:, 1]
    if len(MT) == 0: #Boundary case
        return T
    nodes = {}
    #Construct all node objects
    root = None
    maxVal = -np.inf
    for idx0 in MT:
        for idx in [idx0] + list(MT[idx0]):
            if not idx in nodes:
                nodes[idx] = MergeNode(X[idx, :])
                nodes[idx].P = PS[idx]
            if y[idx] > maxVal:
                root = nodes[idx]
                maxVal = y[idx]
    T.root = root
    #Create all branches
    for idx in MT:
        for cidx in MT[idx]:
            nodes[idx].addChild(nodes[cidx])
    return T

def mergeTreeFrom1DTimeSeries(x):
    """
    Uses union find to make a merge tree object from the time series x
    (NOTE: This code is pretty general and could work to create merge trees
    on any domain if the neighbor set was updated)
    :param x: 1D array representing the time series
    :return: (Merge Tree dictionary, Persistences dictionary, Persistence diagram)
    """
    #Add points from the bottom up
    N = len(x)
    idx = np.argsort(x)
    idxorder = np.zeros(N)
    idxorder[idx] = np.arange(N)
    UFP = np.arange(N) #Pointer to oldest indices
    UFR = np.arange(N) #Representatives of classes
    I = [] #Persistence diagram
    PS = {} #Persistences for merge tree nodes
    MT = {} #Merge tree
    for i in idx:
        neighbs = set([])
        #Find the oldest representatives of the neighbors that
        #are already alive
        for di in [-1, 1]: #Neighbor set is simply left/right
            if i+di >= 0 and i+di < N:
                if idxorder[i+di] < idxorder[i]:
                    neighbs.add(UFFind(UFP, i+di))
        #If none of this point's neighbors are alive yet, this
        #point will become alive with its own class
        if len(neighbs) == 0:
            continue
        neighbs = [n for n in neighbs]
        #Find the oldest class, merge earlier classes with this class,
        #and record the merge events and birth/death times
        oldestNeighb = neighbs[np.argmin([idxorder[n] for n in neighbs])]
        #No matter, what, the current node becomes part of the
        #oldest class to which it is connected
        UFUnion(UFP, oldestNeighb, i, idxorder)
        if len(neighbs) > 1: #A nontrivial merge
            MT[i] = [UFR[n] for n in neighbs] #Add merge tree children
            for n in neighbs:
                if not (n == oldestNeighb):
                    #Record persistence event
                    I.append([x[n], x[i]])
                    pers = x[i] - x[n]
                    PS[i] = pers
                    PS[n] = pers
                UFUnion(UFP, oldestNeighb, n, idxorder)
            #Change the representative for this class to be the
            #saddle point
            UFR[oldestNeighb] = i
    #Add the essential class
    idx1 = np.argmin(x)
    idx2 = np.argmax(x)
    [b, d] = [x[idx1], x[idx2]]
    I.append([b, d])
    I = np.array(I)
    PS[idx1] = d-b
    PS[idx2] = d-b
    return (MT, PS, I)
