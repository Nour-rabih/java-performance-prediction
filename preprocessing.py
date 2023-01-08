# Author: Nour Rabih
# Date: 11/04/2022
# This program reads PMD analysis and creates a formatted csv file


import csv

#find the nth instance of a substring in a string
def find_nth(string, substring, n):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)


"""
In the lines below the txt file is accessed
Student number and errors are extracted
"""
my_file = open("pmdAnalysis_20_21_2.txt", "r")
content = my_file.readlines()

headers=["grade"]
treeData= {} # tree of trees - key: student number, value: tree of errors
studentIDs=[]
for line in content:
    #extract studentID
    end = line.index('/')
    studentID=line[0:end]

    #extract error
    start2 = find_nth(line, ":", 2)+2
    end2 = find_nth(line, ":", 3)
    error= line[start2:end2]

    if error not in headers:
        headers.append(error)
    if studentID not in studentIDs:
        studentIDs.append(studentID)
        treeData[studentID]= {error: 1}
    else:
        if treeData.get(studentID).get(error):
            treeData.get(studentID)[error]+=1
        else:
            treeData.get(studentID)[error]=1

#if a student doesnt have a specific error, assign a 0 to it
for error in headers:
    for student in studentIDs:
        if not treeData.get(student).get(error):
            treeData.get(student)[error]= 0

#getting the grades and assigning them to the students
file = open("finalMarksv8.csv")
csvreader = csv.reader(file)
header = next(csvreader)
rows = []
grade =0
for row in csvreader:
    if int(row[2])>= 40:
        grade= 1
    else:
        grade = 0
    if treeData.get(row[0] ): # if we have the errors of this student
        treeData.get(row[0])['grade']= grade

file.close()

#writing to file
csvData=treeData.values()
with open('errors2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(csvData)
