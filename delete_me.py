class Student(object):
    def __init__(self, name) -> None:
        self.name = name

#========= Testing the above class
# Let's create an object from it
student1 = Student('Ali')
print(student1.name)
student2 = Student('Josh')
print(student2.name)