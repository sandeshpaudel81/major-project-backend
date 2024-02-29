import re
def postpro(key, value):
    if(key=='name'):
        lineSplit = value.split('\n')[0]
        return lineSplit
    elif(key=='dateOfBirth'):
        lineSplit = value.split('\n')[0]
        return lineSplit
    elif(key=='permanentAddressDistrict'):
        lineSplit = value.split('\n')[0]
        colonSplit = re.split('[:\-]', lineSplit)[-1]
        return colonSplit
    elif(key=='placeOfBirthDistrict'):
        lineSplit = value.split('\n')[0]
        colonSplit = re.split('[:\-]', lineSplit)[-1]
        return colonSplit
    elif(key=='citizenshipNumber'):
        lineSplit = value.split('\n')[0]
        colonSplit = lineSplit.split(':')[-1]
        return colonSplit
    elif(key=='issuingDistrict'):
        return value
    elif(key=='gender'):
        colonSplit = value.split(':')[-1]
        lineSplit = colonSplit.split('\n')[0]
        spaceSplit = lineSplit.split()[0]
        return spaceSplit
    elif(key=='placeOfBirthWard'):
        colonSplit = re.split('[:\-]', value)[-1]
        lineSplit = colonSplit.split('\n')[0]
        return lineSplit
    elif(key=='permanentAddressWard'):
        colonSplit = re.split('[:\-]', value)[-1]
        lineSplit = colonSplit.split('\n')[0]
        return lineSplit
    elif(key=='fatherName'):
        lineSplit = value.split('\n')[0]
        return lineSplit
    elif(key=='motherName'):
        lineSplit = value.split('\n')[0]
        return lineSplit
    elif(key=='spouseName'):
        return value
    elif(key=='permanentAddressNagarpalika'):
        colonSplit = re.split('[:\-]', value)[-1]
        lineSplit = colonSplit.split('\n')[0]
        return lineSplit
    elif(key=='placeOfBirthNagarpalika'):
        colonSplit = re.split('[:\-]', value)[-1]
        lineSplit = colonSplit.split('\n')[0]
        return lineSplit
    elif(key=='type'):
        lineSplit = value.split('\n')[0]
        return lineSplit
    elif(key=='nameOfOfficer'):
        lineSplit = value.split('\n')[0]
        return lineSplit
    elif(key=='nameOfOfficer'):
        lineSplit = value.split('\n')[0]
        return lineSplit
    elif(key=='dateofissue'):
        lineSplit = value.split('\n')[0]
        return lineSplit
    else:
        return ''
    
