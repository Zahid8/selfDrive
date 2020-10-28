
def wheelmsgmaker(speed, angle, gear):
    leftspeedstring = "000"
    rightspeedstring = "000"
    leftsign = 0
    rightsign = 0

    if angle > 30:
        angle = 30
    if angle < -30:
        angle = -30
    if angle == 0:
        if gear == 1:
            leftsign = 0
            rightsign = 0
        elif gear == 0:
            leftsign = 1
            rightsign = 1
        if 0 <= speed < 10:
            leftspeedstring = '00' + str(speed)
        elif 10 <= speed < 100:
            leftspeedstring = '0' + str(speed)
        elif speed >= 100:
            leftspeedstring = str(speed)
        if 0 <= speed < 10:
            rightspeedstring = '00' + str(speed)
        elif 10 <= speed < 100:
            rightspeedstring = '0' + str(speed)
        elif speed >= 100:
            rightspeedstring = str(speed)
        msg = '<1' + str(leftsign) + leftspeedstring + str(rightsign) + rightspeedstring + '>'
    elif angle < 0:
        if gear == 1:
            leftsign = 0
            rightsign = 0
        elif gear == 0:
            leftsign = 1
            rightsign = 1
        leftturn = speed - abs(angle)
        if 0 <= leftturn < 10:
            leftspeedstring = '00' + str(leftturn)
        elif 10 <= leftturn < 100:
            leftspeedstring = '0' + str(leftturn)
        elif leftturn >= 100:
            leftspeedstring = str(leftturn)
        rightturn = speed
        if 0 <= rightturn < 10:
            rightspeedstring = '00' + str(rightturn)
        elif 10 <= rightturn < 100:
            rightspeedstring = '0' + str(rightturn)
        elif rightturn >= 100:
            rightspeedstring = str(rightturn)
        msg = '<1' + str(leftsign) + leftspeedstring + str(rightsign) + rightspeedstring + '>'
    else:
        if gear == 1:
            leftsign = 0
            rightsign = 0
        elif gear == 0:
            leftsign = 1
            rightsign = 1
        leftturn = speed
        if 0 <= leftturn < 10:
            leftspeedstring = '00' + str(leftturn)
        elif 10 <= leftturn < 100:
            leftspeedstring = '0' + str(leftturn)
        elif leftturn >= 100:
            leftspeedstring = str(leftturn)
        rightturn = speed - angle
        if 0 <= rightturn < 10:
            rightspeedstring = '00' + str(rightturn)
        elif 10 <= rightturn < 100:
            rightspeedstring = '0' + str(rightturn)
        elif rightturn >= 100:
            rightspeedstring = str(rightturn)
        msg = '<1' + str(leftsign) + leftspeedstring + str(rightsign) + rightspeedstring + '>'
    return msg
