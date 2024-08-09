def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0
def asteroidCollision(asteroids):
    mainStack = []
    for asteroid in asteroids:
        notDestroyed=True
        while(notDestroyed):
            # Check if it is the same direction
            if len(mainStack)==0:
                mainStack.append(asteroid)
                break
            sign1 = sign(mainStack[-1]) 
            sign2 = sign(asteroid)
            if sign1==sign2 or (sign1==-1 and sign2==1):
                mainStack.append(asteroid)
                notDestroyed=False
            # different directions
            elif abs(asteroid) > abs(mainStack[-1]): #Asteriod is strong
                mainStack.pop()
            elif abs(asteroid) < abs(mainStack[-1]): #Weaker
                notDestroyed=False
            elif  abs(asteroid) == abs(mainStack[-1]): #same strength
                mainStack.pop()
                notDestroyed=False
    return mainStack

lol = [5,10,-5]
result = asteroidCollision([5,10,-5])
print(result)

result = asteroidCollision([8,-8])
print(result)

result = asteroidCollision([10,2,-5])
print(result)

result = asteroidCollision([4,2,-5])
print(result)

result = asteroidCollision([-2,-1,1,2])
print(result)