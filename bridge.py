from abc import ABCMeta,abstractmethod

class Shape(metaclass = ABCMeta):
    def __init__(self,color):
        self.color  = color

    @abstractmethod
    def draw(self):
        pass

class Color(metaclass = ABCMeta):
    @abstractmethod
    def paint(self,shape):
        pass

class Line(Shape):
    name = '直线'
    def draw(self):
        self.color.paint(self)

class Circle(Shape):
    name = '圆形'
    def draw(self):
        self.color.paint(self)

class Square(Shape):
    name = '正方形'
    def draw(self):
        self.color.paint(self)

class Red(Color):
    def paint(self, shape):
        print(f'红色的{shape.name}')

class Green(Color):
    def paint(self, shape):
        print(f'绿色的{shape.name}')

class Yellow(Color):
    def paint(self, shape):
        print(f'黄色的{shape.name}')

p = Line(Red())
p.draw()
p2 = Circle(Green())
p2.draw()
p3 =  Square(Yellow())
p3.draw()
