from abc import ABCMeta, abstractmethod

# 发布者


class Publish:
    def __init__(self) -> None:
        self._obs = []

    def attach(self, obs):
        self._obs.append(obs)

    def dettach(self, obs):
        if obs in self._obs:
            self._obs.remove(obs)

    def notify(self):
        for obs in self._obs:
            obs.update(self)


# 订阅者
class Observer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, publish):
        pass

# 具体发布者
class StaffPublish(Publish):
    def __init__(self, company_info=None) -> None:
        super().__init__()
        self._company_info = company_info

    @property
    def company_info(self):
        return self._company_info

    @company_info.setter
    def company_info(self, info):
        self._company_info = info
        self.notify()

# 具体订阅者
class StaffObserver(Observer):
    def __init__(self) -> None:
        self.info = []

    def update(self, publish):
        self.info.append(publish.company_info)

p = StaffPublish("公司基本信息")
o1 = StaffObserver()
o2 = StaffObserver()
p.attach(o1)
p.attach(o2)
print(o1.info)
print(o2.info)
p.company_info = "重大通知0218"
p.company_info = "重大通知0329"
print(o1.info)
print(o2.info)
p.dettach(o1)

p.company_info = "重大通知0330"
print(o1.info)
print(o2.info)