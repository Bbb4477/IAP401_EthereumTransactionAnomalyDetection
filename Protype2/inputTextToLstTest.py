f = open("InputNoteTest.txt","r")
for i in f:
    print(i[:len(i)-1],end=",")