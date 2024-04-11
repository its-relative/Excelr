import numpy as np
while True:
    random = []
    for i in range(3):
        random.append(np.random.randint(100,200))
    random

    print("Enter the multiplication fo all these: ", random[0], random[1], random[2])
    n = int(input())
    if n == random[0]*random[1]*random[2]:
        print("Bravo")
    else:
        print("Not correct")
    print("lets go again")

    if n == "q":
        break