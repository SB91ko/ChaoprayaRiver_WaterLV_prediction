class test_class:
    def __init__(self,txt,num=1,flag=False):
        self.txt = txt
        self.num = num
        self.flag = flag

        self.show_com(txt)

    def show_com(self,txt,num,flag):
        print("your text is ......",self.txt)
        if self.flag:
            print("number power is ....",self.power_num(self.num))

    def power_num(self,your_num):
        return your_num*your_num

class basic:
    def __init__(self,basictask,t):
        self.basictask = basictask
        self.t = t
        self.basic_print()
    def basic_print(self):
        print(self.basictask*self.t)

class Person:
    def __init__(self, firstName, lastName):
        self.firstName = firstName
        self.lastName = lastName
        self.getName()
    def getName(self):
        return self.firstName + ' ' + self.lastName



basic("test",5)
print("*"*50)
#test_class('test txt',2,True)
##################
p = Person('song','bun')

#################
print(p)
