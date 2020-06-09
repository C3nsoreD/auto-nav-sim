import turtle
from random import randint
"""
Review of turtle graphics 
"""


size = 20
circles = 20
turtle.speed(10)

turtle.colormode(255)
# move function 
def move(length, angle):
    turtle.right(angle)
    turtle.forward(length)

def square():
    turtle.pendown()
    turtle.color(randint(0,255), randint(0,255), randint(0,255))
    turtle.begin_fill()
    for i in range(4):
        move(size,-90)
    turtle.end_fill()
    turtle.penup()

# start
turtle.penup()

for circle in range (circles):
    if circle == 0:
        square()
        move(size,-90)
        move(size,-90)
        move(size,-90)
        move(0,180)
    for i in range (4):
        move(0,90)
        for j in range (circle+1):
            square()
            move(size,-90)
            move(size,90)
        move(-size,0)
    move(-size,90)
    move(size,-180)
    move(0,90)

turtle.exitonclick()
