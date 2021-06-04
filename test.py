class Test:
    i = 10


myObj = Test()

myObj.i = 5
Test.i = 15
print(myObj.i)
print(Test.i)
