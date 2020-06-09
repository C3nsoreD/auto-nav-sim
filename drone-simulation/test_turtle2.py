""" Application to run """
from turtle import Turtle, Screen

# Setup Screen
wn = Screen()
wn.setup(700, 700)
wn.title('white')
wn.bgcolor('black')

# Create Player
player = Turtle('triangle')
player.speed('slowest')
player.color('white')
player.penup()

t = Turtle()
t.color("red")
t.penup()
t.goto(50, 50)
t.pendown()
t.hideturtle()

def forward():
    player.forward(20)

def left():
    player.left(90)

def right():
    player.right(90)

def square(length):
    for steps in range(4):
        t.forward(length)
        t.left(90)
square(50)

wn.onkey(forward, 'Up')
wn.onkey(left, 'Left')
wn.onkey(right, 'Right')

wn.listen()
wn.mainloop()
