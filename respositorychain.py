from abc import ABCMeta,abstractmethod
from cgitb import handler

class Handler(metaclass = ABCMeta):
    @abstractmethod
    def handle(self,day):
        pass

class ManagerHandler(Handler):
    def handle(self, day):
        if day<=10:
            print('总经理批准')
        else:
            print('总经理不批')

class DepartHandler(Handler):
    def __init__(self) -> None:
        self.next = ManagerHandler()

    def handle(self,day):
        if day < 5:
            print('部门领导批准')
        else:
            self.next.handle(day)

class DirectorHandler(Handler):
    def __init__(self) -> None:
        self.next = DepartHandler()

    def handle(self, day):
        if day<=3:
            print('直属领导批假')
        else:
            self.next.handle(day)


d = DirectorHandler()
for day in range(1,10):
    print(f'{day}天-----')
    d.handle(day)
    print('-----')

