import sys; args = sys.argv[1:]
import time

class PQElement:
    def __init__(self, state, path, g, h):
        self.state = state
        self.path = path
        self.g = g
        self.h = h
        self.f = g + h
    def __lt__(self, other):
        return self.f < other.f
    def __str__(self) -> str:
        return self.state

class QueueNode:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class Queue:
    def __init__(self):
        self.front = None
        self.rear = None

    def isEmpty(self): return self.front is None

    def push(self, data):
        newNode = QueueNode(data)
        if self.rear is None: self.front = self.rear = newNode
        else:
            self.rear.next = newNode
            self.rear = newNode

    def pop(self):
        if self.isEmpty(): return None
        data = self.front.data
        self.front = self.front.next
        if self.front is None: self.rear = None
        return data
    def __str__(self) -> str:
        res = "["
        current = self.front
        while current is not None:
            res += str(current.data) + ", "
            current = current.next
        if len(res) == 1: return "[]"
        return res[:-2] + "]"

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def parent(self, index):
        return (index - 1) // 2

    def leftChild(self, index):
        return 2 * index + 1

    def rightChild(self, index):
        return 2 * index + 2

    def heapUp(self, index):
        p = self.parent(index)
        if index > 0 and self.heap[index] < self.heap[p]:
            self.heap[index], self.heap[p] = self.heap[p], self.heap[index]
            self.heapUp(p)

    def heapDown(self, index):
        l = self.leftChild(index)
        r = self.rightChild(index)
        mi = index
        if l < len(self.heap) and self.heap[l] < self.heap[mi]:
            mi = l
        if r < len(self.heap) and self.heap[r] < self.heap[mi]:
            mi = r
        if mi != index:
            self.heap[index], self.heap[mi] = self.heap[mi], self.heap[index]
            self.heapDown(mi)

    def push(self, value):
        self.heap.append(value)
        self.heapUp(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapDown(0)
        return root

    def peek(self):
        return self.heap[0] if self.heap else None

    def isEmpty(self):
        return len(self.heap) == 0
  
def dist_heuristic(state, goal): #O(n)
    s = 0
    for c in state:
        if c == "_": continue
        sind, gind = state.index(c), goal.index(c)
        s += abs(sind//4 - gind//4) + abs(sind%4 - gind%4)
    return s

def inversion_count(start, end): #O(n^2)
   s = "".join("ABCDEFGHIJKLMNOP"[end.index(c)] if c != "_" else "_" for c in start)
   nums = list(s[:(j:=s.index("_"))] + s[j + 1:])
   curS = sum(sum(a < n for a in nums[i+1:]) for (i, n) in enumerate(nums))
   dRow = abs(start.index("_") // 4 - end.index("_") // 4)
   if dRow % 2 == 1: curS += 1
   return curS

def generate_children(state, o): #O(1)
   w, h = 4, 4
   u = state.index("_")
   res = []
   if u >= w: res += [(swap(state, u - w, u), "D" if o else "U", state[u - w], u - w, u)] #Up
   if u < w*h - w:res += [(swap(state, u, u + w), "U" if o else "D", state[u + w], u + w, u)] #Down
   if u%w != 0: res += [(swap(state, u - 1, u), "R" if o else "L", state[u - 1], u - 1, u)] #Left
   if u%w != w - 1: res += [(swap(state, u, u + 1), "L" if o else "R", state[u + 1], u + 1, u)] #Right
   return res

def changeInHeuristic(goalState, movedTile, pi, ci):
    gi = goalState.index(movedTile)
    return abs(ci // 4 - gi // 4) + abs(ci % 4 - gi % 4) - abs(pi // 4 - gi // 4) - abs(pi % 4 - gi % 4)
    

def a_star(start, goal, startTime, puzzleIndex):
    if start == goal: return True, ""
    if inversion_count(initial, goal) % 2 == 1: return (False, "")
    pq = PriorityQueue()
    pq.push(PQElement(start, "", 0, dist_heuristic(start, goal)))

    visited = set()

    while not pq.isEmpty():
        if time.time() - startTime > 5: return bastar(start, goal, startTime, puzzleIndex)
        popped = pq.pop()
        current, path, g, h = popped.state, popped.path, popped.g, popped.h
        if current in visited: continue
        if current == goal: return (True, path)
        visited.add(current)
        for neighbor, move, tile, pi, ci in generate_children(current, False):
            if neighbor in visited: continue
            pq.push(PQElement(neighbor, path + move, g + 1, h + changeInHeuristic(goal, tile, pi, ci)))
    return (False, "")

puzzles = open(args[0]).read().splitlines()

index = 0
goal = puzzles[0]
oStart = time.time()
for initial in puzzles:
    width = int(len(goal)**(1/2))
    height = width
    start = time.time()
    success, result = a_star(initial, goal, start, index)
    end = time.time()
    if not success:
        print(str(index) + ": " + initial + " X")
    else:
        if len(result) == 0:
            print(str(index) + ": " + initial + " G")
        else:
            print(str(index) + ": " + initial +  " len " + str(len(result) + 1) + " in " + str(f"{end - start:.2f}" + "s: " + result))
    index += 1
# Armaan Ahmed, Period 6, 2026