def minimum_edit_distance(string1, string2):
    # Base Case (far-left column and bottom row of distance matrix)
    if string1 == '#' and string2 == '#':
        return 0
    elif string2 == '#':
        return len(string1) - 1
    elif string1 == '#':
        return len(string2) - 1

    # Recursive calls to determine the deletion, insertion, and substitution costs.
    deletion = minimum_edit_distance(string1[:-1], string2) + deletion_cost
    insertion = minimum_edit_distance(string1, string2[:-1]) + insertion_cost
    substitution = \
        minimum_edit_distance(string1[:-1], string2[:-1]) if string1[-1] == string2[-1] \
        else minimum_edit_distance(string1[:-1], string2[:-1]) + substitution_cost

    # Find the best path
    return min(deletion, insertion, substitution)


# Input the costs
deletion_cost = int(input("Enter the deletion cost: "))
insertion_cost = int(input("Enter the insertion cost: "))
substitution_cost = int(input("Enter the substitution cost: "))

# Input the two strings
s1 = input("Enter String 1: ")
s2 = input("Enter String 2: ")

# Final output of the minimum edit distance.
print(f'Mimimum edit distance for {s1} and {s2}: {minimum_edit_distance("#" + s1, "#" + s2)}')
