f = open("t", "r")

c = 149
new_f = open("t_2", "w")

for line in f.readlines():

    # addend = f'fl_tech_logo_{c} DB "'
    # line_2 = line
    # line_2 = line_2.split("\n")[0]
    #
    # s = addend + line_2 + '", 0Dh, 0Ah, 0\n'
    # # print(s)
    # new_f.write(s)

    print(f"mov edx, OFFSET fl_tech_logo_{c}")
    print("Call WriteString")
    c += 1
