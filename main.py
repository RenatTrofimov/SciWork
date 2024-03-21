import uuid
import math
import matplotlib.pyplot as plt
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

class Coordinates:
    x = None
    y = None
    z = None
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

class FigureGetter:
    def get(self, node):
        pass
class UpperGetter(FigureGetter):
    def get(self, node):
        pass
class HexUpLeftGetter(FigureGetter):
    def get(self, node):
        return node.nodes[4]
class HexUpRightGetter(FigureGetter):
    def get(self, node):
        return node.nodes[5]       
class HexDownLeftGetter(FigureGetter):
    def get(self, node):
        return node.nodes[2]
class HexDownRightGetter(FigureGetter):
    def get(self, node):
        return node.nodes[1]       
class HexLeftGetter(FigureGetter):
    def get(self, node):
        return node.nodes[3]
class HexRightGetter(FigureGetter):
    def get(self, node):
        return node.nodes[0]
class AllNeighGetter(FigureGetter):
    def get(self, node):
        return node.neighbour   
class HexUpNeighGetter(FigureGetter):
    def get(self, node):
        return node.neighbour[0]   
class HexDownNeighGetter(FigureGetter):
    def get(self, node):
        return node.neighbour[3] 

class HexUpRightNeighGetter(FigureGetter):
    def get(self, node):
        return node.neighbour[5]   
class HexDownRightNeighGetter(FigureGetter):
    def get(self, node):
        return node.neighbour[4]

class HexUpLeftNeighGetter(FigureGetter):
    def get(self, node):
        return node.neighbour[1]   
class HexDownLeftNeighGetter(FigureGetter):
    def get(self, node):
        return node.neighbour[2]


class Handle:
    def execute(self):
        pass
class AddNeigh(Handle):
    def __init__(self, whatNode, whichNode):
        self.whatNode = whatNode
        self.whichNode = whichNode
    def execute(self):
        self.whatNode.neighbour.append(self.whichNode)
class RemoveAndReplaceNode(Handle):
    def __init__(self, upNode, downNode):
        self.upNode = upNode
        self.downNode = downNode

class HexColDownReplace(RemoveAndReplaceNode):
    def execute(self):
        if not self.upNode.get(HexDownLeftGetter()) is self.downNode.get(HexUpLeftGetter()):
            self.upNode.action(Replacer(self.upNode.get(HexDownLeftGetter()), self.downNode.get(HexUpLeftGetter())))
        if not self.upNode.get(HexDownRightGetter()) is self.downNode.get(HexUpRightGetter()):
            self.upNode.action(Replacer(self.upNode.get(HexDownRightGetter()), self.downNode.get(HexUpRightGetter())))
        
class NeighbourSetuper:
    def addNeightbour(self,firstNode, secondNode):
        pass

class SimpleNeighbourSetuper(NeighbourSetuper):
    def addNeightbour(self,firstNode, secondNode):
        firstNode.getNeighbour().append(secondNode)
        secondNode.getNeighbour().append(firstNode)
class HexFigUp(NeighbourSetuper):
    def addNeightbour(self,firstNode, secondNode):
        firstNode.setUp(secondNode)
        secondNode.setDown(firstNode)
class HexFigLeft(NeighbourSetuper):
    def addNeightbour(self,firstNode, secondNode):
        firstNode.setLeft(secondNode)
        secondNode.setRight(firstNode)
class HexFigRight(NeighbourSetuper):

    def addNeightbour(self,firstNode, secondNode):
        firstNode.setRight(secondNode)
        secondNode.setLeft(firstNode)
class HexFigDown(NeighbourSetuper):
    def addNeightbour(self,firstNode, secondNode):
        firstNode.setDown(secondNode)
        secondNode.setUp(firstNode)
class HexColLeft(NeighbourSetuper):
    def addNeightbour(self,firstNode, secondNode):
        firstNode.getNeighbour()[0] = secondNode
        secondNode.getNeighbour()[1] = firstNode
class HexColRight(NeighbourSetuper):
    def addNeightbour(self,firstNode, secondNode):
        firstNode.getNeighbour()[1] = secondNode
        secondNode.getNeighbour()[0] = firstNode
class Connector:
    def connect(self, firstNode, secondNode):
        pass
class NeighbourRemover:
    def addNeightbour(self,firstNode, secondNode):
        pass
class Setter:
    def setup(self, aNode):
        pass
class SetNeighByIndex(Setter):
    def __init__(self, index):
        self.index = index
    def setup(self, orgNode, aNode):
        orgNode.get(AllNeighGetter())[self.index] = aNode
class SetNeighToLastPos(Setter):
    def setup(self, orgNode, aNode):
        orgNode.get(AllNeighGetter()).append(aNode)

class Replacer(Handle):
    def __init__(self, oldNode, newNode):
        self.oldNode = oldNode
        self.newNode = newNode
    def execute(self):
        neig = self.oldNode.get(AllNeighGetter())
        for i in range(len(neig)):
            for j, node in enumerate(neig[i].get(AllNeighGetter())):
                if node is self.oldNode:
                    neig[i].setup(SetNeighByIndex(j), self.newNode)
                    self.newNode.setup(SetNeighToLastPos(), neig[i])
            
            

class AutoInc:
    inc = -1
    def getId(self):
        self.inc+=1
        return self.inc           
                    
AI = AutoInc()

class Node :
    nId = None
    coords = None
    neighbour = None   
    def __init__(self,x = 0,y = 0):
        self.nId = uuid.uuid4()
        self.neighbour = []
        self.coords = Coordinates()
        self.coords.x = x
        self.coords.y = y
    def setCoords(self, coords):
        self.coords = coords
    def getCoords(self):
        return self.coords
    def draw(self, drawer):
        drawer.draw(self)
    def addNeightbour(self, neighbourSetuper, node):
        neighbourSetuper.addNeightbour(self, node)
    def removeNeightbour(self, neighbourRemover, node):
        neighbourRemover.removeNeightbour(self, node)
    
    def getNeighbour(self):
        return self.neighbour
    def setNeighbour(self, neighbour):
        self.neighbour = neighbour
    def setLeft(self, node):
        pass
    def setRight(self, node):
        pass
    def setUp(self, node):
        pass
    def setDown(self, node):
        pass
    def getLeft(self):
        pass
    def getRight(self):
        pass
    def getUp(self):
        pass
    def getDown(self):
        pass
    def toUp(self):
        pass
    def toDown(self):
        pass
    def toLeft(self):
        pass
    def toRight(self):
        pass
    def get(self, getter):
        return getter.get(self)
    def setup(self, setter , node):
        return setter.setup(self, node)
    def remove(self, remover):
        return remover.remove()
    def multiply(self, multiplayer):
        multiplayer.action()
    def action(self, handler):
        handler.execute()
    
class Figure(Node):
    nodes = []
    def draw(self, drawer):
        for node in self.nodes:
            if node is not None:
                node.draw(drawer)
    def addNode(self, node):
        self.nodes.append(node)
    def setNodes(self, nodes):
        self.nodes = nodes
    def initNodes(self):
        pass
class HexFig(Figure):
    radius = 25
    def __init__(self, x,y):
        super().__init__(x,y)
        self.neighbour = [None,None,None,None,None,None]
        self.nodes = [None,None,None,None,None,None]
        self.initNodes()
    def initNodes(self):
        for index in range(6):
            if self.nodes[index] is None:
                aNode = Node(self.coords.x + self.radius*math.cos(math.pi*index/3), self.coords.y + self.radius*math.sin(math.pi*index/3))
                self.nodes[index] = aNode
                #self.nodes[index].nId = index 
        for index in range(6):
            self.nodes[index].addNeightbour(SimpleNeighbourSetuper(), self.nodes[(index+1)%6])
    def toDown(self):
        pass
    def toUp(self):
        pass
    def setUp(self, node):
        self.neighbour[0] = node
    def setDown(self, node):
        self.neighbour[3] = node
    def getUp(self):
        return self.neighbour[0]
    def getDown(self):
        return self.neighbour[3]
class HexUpSide(HexFig):
    def toUp(self):
        aNode = HexUpSide(self.coords.x, self.coords.y - math.sqrt(3)*self.radius)
        aNode.setNodes([None,self.nodes[5],self.nodes[4],None,None,None])
        aNode.initNodes()
        self.addNeightbour(HexFigUp(), aNode)
    def toDown(self):
        aNode = HexUpSide(self.coords.x, self.coords.y + math.sqrt(3)*self.radius)
        aNode.setNodes([None,None,None,None,self.nodes[2],self.nodes[1]])
        aNode.initNodes()
        self.addNeightbour(HexFigDown(), aNode)
    def toLeft(self):
        angle = 4*math.pi/3
        koeff = math.cos(math.pi/6)
        aNode = HexDownSide(
                self.coords.x + koeff*2*self.radius*math.sin(angle),
                self.coords.y + koeff*2*self.radius*math.cos(angle)
                )
        node = self.get(HexUpNeighGetter())
        if not (node is None):
            node = node.get(HexLeftGetter())
        aNode.setNodes([self.nodes[4],self.nodes[3],None,None,None,node])
        aNode.initNodes()
        self.addNeightbour(
            HexFigLeft(), aNode)
    def toRight(self):
        angle = 2*math.pi/3
        koeff = math.cos(math.pi/6)
        aNode = HexDownSide(
                self.coords.x + koeff*2*self.radius*math.sin(angle),
                self.coords.y + koeff*2*self.radius*math.cos(angle)
                )
        node = self.get(HexUpNeighGetter())
        if not (node is None):
            node = node.get(HexRightGetter())
        aNode.setNodes([None,None,self.nodes[0],self.nodes[5],node,None])
        aNode.initNodes()
        self.addNeightbour(HexFigRight(), aNode)
        
    def setLeft(self, node):
        self.getNeighbour()[5] = node
    def getLeft(self):
        return self.getNeighbour()[5]
    def getRight(self):
        return self.getNeighbour()[1]
    def setRight(self, node):
        self.getNeighbour()[1] = node
class HexDownSide(HexFig):
    def toUp(self):
        aNode = HexDownSide(self.coords.x, self.coords.y - math.sqrt(3)*self.radius)
        aNode.setNodes([None,self.nodes[5],self.nodes[4],None,None,None])
        aNode.initNodes()
        self.addNeightbour(HexFigUp(), aNode)
    def toDown(self):
        aNode = HexDownSide(self.coords.x, self.coords.y + math.sqrt(3)*self.radius)
        aNode.setNodes([None,None,None,None,self.nodes[2],self.nodes[1]])
        aNode.initNodes()
        self.addNeightbour(HexFigDown(), aNode)
    def toLeft(self):
        angle = 5*math.pi/3
        koeff = math.cos(math.pi/6)
        aNode = HexUpSide(
                self.coords.x + koeff*2*self.radius*math.sin(angle),
                self.coords.y + koeff*2*self.radius*math.cos(angle)
                )
        node = self.get(HexDownNeighGetter())
        if not (node is None):
            node = node.get(HexLeftGetter())
        aNode.setNodes([self.nodes[2], node,None,None,None,self.nodes[3]])
        aNode.initNodes()
        self.addNeightbour(
            HexFigLeft(), aNode)
    def toRight(self):
        angle = 1*math.pi/3
        koeff = math.cos(math.pi/6)
        aNode = HexUpSide(
                self.coords.x + koeff*2*self.radius*math.sin(angle),
                self.coords.y + koeff*2*self.radius*math.cos(angle)
                )
        node = self.get(HexDownNeighGetter())
        if not (node is None):
            node = node.get(HexRightGetter())
        aNode.setNodes([None,None,node,self.nodes[1],self.nodes[0],None])
        aNode.initNodes()
        self.addNeightbour(
            HexFigRight(), aNode)
    def setLeft(self, node):
        self.getNeighbour()[4] = node
    def getLeft(self):
        return self.getNeighbour()[4]
    def getRight(self):
        return self.getNeighbour()[2]
    def setRight(self, node):
        self.getNeighbour()[2] = node
class HexColm(Figure):
    def __init__(self, x,y):
        super().__init__(x,y)
        self.nodes = []
        self.neighbour = [None, None]
        self.nodes.append(HexDownSide(x,y))
    def toUp(self):
        self.nodes[0].toUp()
        self.nodes.insert(0, self.nodes[0].getNeighbour()[0])
    def toDown(self):
        self.nodes[-1].toDown()
        self.nodes.append(self.nodes[-1].getNeighbour()[3])
    def toRight(self):
        newHexCol = HexColm(0,0)
        newHexCol.setNodes([])
        for node in self.nodes:
            node.toRight()
            newHexCol.addNode(node.getRight())
        self.addNeightbour(HexColRight(), newHexCol)
    def toLeft(self):
        newHexCol = HexColm(0,0)
        newHexCol.setNodes([])
        for node in self.nodes:
            node.toLeft()
            newHexCol.addNode(node.getLeft())
        self.addNeightbour(HexColLeft(), newHexCol)
    def addNode(self, node):
        super().addNode(node)
        if len(self.nodes) > 1:
            self.action(HexColDownReplace(self.nodes[-2], self.nodes[-1]))
    def getLeft(self):
        return self.neighbour[0]
    def getRight(self):
        return self.neighbour[1]
class HexNet(Figure):
    def __init__(self, x,y):
        super().__init__(x,y)
        self.nodes = []
        self.nodes.append(HexColm(x,y))
    def toUp(self):
        for node in self.nodes:
            node.toUp()
    def toDown(self):
        for node in self.nodes:
            node.toDown()
    def toLeft(self):
        self.nodes[0].toLeft()
        self.nodes.insert(0, self.nodes[0].getLeft())
    def toRight(self):
        self.nodes[-1].toRight()
        self.nodes.append(self.nodes[-1].getRight())
class SimpleDrawer:
    def draw(self, node):
        plt.scatter(node.coords.x,node.coords.y)
class SimpleConnectionDrawer:
    usedNodes = None
    def __init__(self, scene):
        self.scene = scene
        self.usedNodes = set()
    def draw(self, node):   
        cords = node.getCoords()
        for neibor in node.getNeighbour():
            if neibor in self.usedNodes or neibor is None :
                continue
            nCords = neibor.getCoords()
            self.scene.addLine(cords.x, cords.y, nCords.x, nCords.y)
        self.scene.addEllipse(cords.x - 5, cords.y - 5, 10, 10)
        self.usedNodes.add(node)
        #io = QtWidgets.QGraphicsTextItem()
        #io.setPos(cords.x - 5, cords.y + 5)
        #io.setPlainText(str(node.nId))
        #self.scene.addItem(io) 
class SimpleSelectionDrawer:
    usedNodes = None
    def __init__(self, scene, coord):
        self.scene = scene
        self.coord = coord
        self.usedNodes = []
    def draw(self, node):
        cords = node.getCoords()
        if math.sqrt((cords.x - self.coord.x())**2 + (cords.y - self.coord.y())**2) < 10:
            for neibor in node.getNeighbour():
                if not neibor is None:
                    nCords = neibor.getCoords()
                    self.scene.addEllipse(nCords.x - 5, nCords.y - 5, 10, 10, brush=QtGui.QBrush(QtCore.Qt.GlobalColor.red))
            self.scene.addEllipse(cords.x - 5, cords.y - 5, 10, 10, brush=QtGui.QBrush(QtCore.Qt.GlobalColor.red))
# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.


# Create a Qt widget, which will be our window.



class Remover:
    def __init__(self, aNode, delNode):
        self.aNode = aNode
        self.delNode = delNode
    def remove(self):
        pass

class NeighRemover(Remover):
    def remove(self):
        for i in range(len(self.aNode.get(AllNeighGetter()))):
            if self.aNode.get(AllNeighGetter())[i] is self.delNode:
                self.aNode.get(AllNeighGetter()).remove(self.delNode)
                break
                

class SimpleSelectionRemover:
    usedNodes = None
    def __init__(self, coord):
        self.coord = coord
        self.usedNodes = []
    def draw(self, node):
        
        cords = node.getCoords()
        if math.sqrt((cords.x - self.coord.x())**2 + (cords.y - self.coord.y())**2) < 10:
            for index, neibor in enumerate(node.getNeighbour()):
                if not neibor is None:
                    neibor.remove(NeighRemover(node.getNeighbour()[index], node))
                    node.getNeighbour()[index] = None
                    
            
        



class Systema:
    def __init__(self):
        self.Nodes = []
        self.radius = 25
        self.coords = Coordinates(25,25)
        k,j = 1,1
        for row in range (0,17):
            Nodes = []
            lx = self.coords.x
            ly = self.coords.y
            for col in range(0,17):
                for index in range(6):
                    aNode = Node(
                            lx + self.radius*math.cos(math.pi*index/3), 
                            ly + self.radius*math.sin(math.pi*index/3))
                    Nodes.append(aNode)
                lx += round(2*self.radius*math.cos(math.pi/6)*math.sqrt(3)/2, 6)
                ly += round(((-1)**col)*2*j*self.radius*math.sin(math.pi/6)*math.sqrt(3)/2, 6)
            self.coords.y += round(2*self.radius*math.sin(math.pi/6)*math.sqrt(3), 6)
            self.Nodes += Nodes
        UniqueCoords = []
        UniqueNodes = []
        for node in self.Nodes:
            if not (round(node.coords.x, 6), round(node.coords.y, 6))  in  UniqueCoords :
                UniqueCoords.append((round(node.coords.x, 6), round(node.coords.y, 6)))
                UniqueNodes.append(node)
        #print(UniqueCoords)
        #print(UniqueNodes)
        #Образование связей
        for index, coords in enumerate (UniqueCoords):
            """for i in range(3):
                x = coords[0] + 2*self.radius*math.cos(math.pi/6)*math.cos(i*math.pi/6 + math.pi/6)
                y = coords[1] + 2*self.radius*math.cos(math.pi/6)*math.sin(i*math.pi/6 + math.pi/6)
                x = round(x, 6)
                y = round(y, 6)
                if (x,y) in UniqueCoords:
                    #print(index,UniqueCoords.index([x,y]))
                    UniqueNodes[index].action(AddNeigh(UniqueNodes[index], UniqueNodes[UniqueCoords.index((x,y))]))
                x = coords[0] - 2*self.radius*math.cos(math.pi/6)*math.cos(i*math.pi/6 + math.pi/6)
                y = coords[1] - 2*self.radius*math.cos(math.pi/6)*math.sin(i*math.pi/6 + math.pi/6)
                x = round(x, 6)
                y = round(y, 6)
                if (x,y) in UniqueCoords:
                    #print(index,UniqueCoords.index([x,y]))
                    UniqueNodes[index].action(AddNeigh(UniqueNodes[index], UniqueNodes[UniqueCoords.index((x,y))]))
                x = coords[0] + 2*self.radius*math.cos(math.pi/6)*math.cos(i*math.pi/6 - math.pi/6)
                y = coords[1] + 2*self.radius*math.cos(math.pi/6)*math.sin(i*math.pi/6 - math.pi/6)
                x = round(x, 6)
                y = round(y, 6)
                if (x,y) in UniqueCoords:
                    #print(index,UniqueCoords.index([x,y]))
                    UniqueNodes[index].action(AddNeigh(UniqueNodes[index], UniqueNodes[UniqueCoords.index((x,y))]))
                x = coords[0] - 2*self.radius*math.cos(math.pi/6)*math.cos(i*math.pi/6 - math.pi/6)
                y = coords[1] - 2*self.radius*math.cos(math.pi/6)*math.sin(i*math.pi/6 - math.pi/6)
                x = round(x, 6)
                y = round(y, 6)
                if (x,y) in UniqueCoords:
                    #print(index,UniqueCoords.index([x,y]))
                    UniqueNodes[index].action(AddNeigh(UniqueNodes[index], UniqueNodes[UniqueCoords.index((x,y))]))"""
            
            for i in range(3):
                x = coords[0] + self.radius*math.cos(i*2*math.pi/3)
                y = coords[1] + self.radius*math.sin(i*2*math.pi/3)
                x = round(x, 6)
                y = round(y, 6)
                if (x,y) in UniqueCoords:
                    #print(index,UniqueCoords.index([x,y]))
                    UniqueNodes[index].action(AddNeigh(UniqueNodes[index], UniqueNodes[UniqueCoords.index((x,y))]))
                x = coords[0] - self.radius*math.cos(i*2*math.pi/3)
                y = coords[1] - self.radius*math.sin(i*2*math.pi/3)
                x = round(x, 6)
                y = round(y, 6)
                if (x,y) in UniqueCoords:
                    #print(index,UniqueCoords.index([x,y]))
                    UniqueNodes[index].action(AddNeigh(UniqueNodes[index], UniqueNodes[UniqueCoords.index((x,y))]))
        self.Nodes = UniqueNodes
    def output(self):
        Nodes = []
        Coords = []
        for index, node in enumerate(self.Nodes):
            self.Nodes[index].nId = index
        for node in self.Nodes:
            #print(str(node.nId) + ": " + str([i.nId for i in node.neighbour]))
            Nodes.append(list(dict.fromkeys([i.nId for i in node.neighbour])))
            Coords.append([node.coords.x, node.coords.y])
        out = np.zeros((len(Nodes), np.max([len(node) for node in Nodes])))
        for i in range(len(Nodes)):
            out[i,:len(Nodes[i])] = Nodes[i]
        print(out)
        Nodes = list(out.astype(int))
        Nodes.insert(0, (len(Nodes)* np.max([len(node) for node in Nodes])))
        with open('aNodes.npy', 'wb') as f:
            np.save(f, np.array(Nodes))
        np.savetxt('aNodes.out', np.array(Nodes),  fmt='%s')
        with open('aCoords.npy', 'wb') as f:
            np.save(f, np.array(Coords))
    def draw(self, drawer):
       
        for node in self.Nodes:
            node.draw(drawer)
           
    def remove(self, coords):
        
        for node in self.Nodes:
            if math.sqrt((coords.x() - node.coords.x)**2 + (coords.y() - node.coords.y)**2) < 10:
                for index, neibor in enumerate(node.getNeighbour()):
                    if not neibor is None:
                        neibor.remove(NeighRemover(node.getNeighbour()[index], node))
                self.Nodes.remove(node)
                break
                
                
            
            
                    
class MyQtNGV(QtWidgets.QGraphicsView):
    def setupScene(self):
        scene = QtWidgets.QGraphicsScene()
        self.systema = Systema()
        
        SD = SimpleDrawer()
        SCD = SimpleConnectionDrawer(scene)
        self.systema.draw(SD)
        self.systema.draw(SCD)
        self.setScene(scene)
        self.update()
        
    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Type.MouseMove:
            if event.buttons() == QtCore.Qt.MouseButton.NoButton:
                pos = event.scenePosition()
                scene = QtWidgets.QGraphicsScene()
                SCD = SimpleConnectionDrawer(scene)
                SSD = SimpleSelectionDrawer(scene, pos)
                self.systema.draw(SCD)
                self.systema.draw(SSD)
                self.setScene(scene)
                self.update()
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.buttons() == QtCore.Qt.MouseButton.LeftButton:
                self.systema.output()
            if event.buttons() == QtCore.Qt.MouseButton.RightButton:
                self.systema.remove(event.position()) 
                scene = QtWidgets.QGraphicsScene()
                SCD = SimpleConnectionDrawer(scene)
                SSD = SimpleSelectionDrawer(scene, event.position())
                self.systema.draw(SCD)
                self.systema.draw(SSD)
                self.setScene(scene)
                self.update()
        return QtWidgets.QGraphicsView.eventFilter(self, source, event)

class MyQtGV(QtWidgets.QGraphicsView):
    def setupScene(self):
        scene = QtWidgets.QGraphicsScene()
        self.hex = HexNet(25,25)
        for i in range(2):
            self.hex.toDown()
        for i in range(2):
            self.hex.toRight()
        
        SD = SimpleDrawer()
        SCD = SimpleConnectionDrawer(scene)
        self.hex.draw(SD)
        self.hex.draw(SCD)
        self.setScene(scene)
        
    def mousePressEvent(self, event):
        
        self.hex.draw(SimpleSelectionRemover(event.position())) 
        scene = QtWidgets.QGraphicsScene()
        SCD = SimpleConnectionDrawer(scene)
        SSD = SimpleSelectionDrawer(scene, event.position())
        self.hex.draw(SCD)
        self.hex.draw(SSD)
        self.setScene(scene)
        self.update()
    
    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Type.MouseMove:
            if event.buttons() == QtCore.Qt.MouseButton.NoButton:
                pos = event.scenePosition()
                scene = QtWidgets.QGraphicsScene()
                SCD = SimpleConnectionDrawer(scene)
                SSD = SimpleSelectionDrawer(scene, pos)
                self.hex.draw(SCD)
                self.hex.draw(SSD)
                self.setScene(scene)
                self.update()
            else:
                pass # do other stuff
        return QtWidgets.QGraphicsView.eventFilter(self, source, event)

app = QtWidgets.QApplication(sys.argv)
window = MyQtNGV()
window.setupScene()
window.installEventFilter(window)
window.show()  # IMPORTANT!!!!! Windows are hidden by default.
app.installEventFilter(window)
# Start the event loop.
app.exec()


