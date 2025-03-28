from collections import defaultdict

# 1. Two Number Sum

def two_sum(arr: list, target: int) -> list:
    '''
        We create a length reference to to the length of the array, in order
        to avoid calling the len method twice.

        We then do a nested for loop to check every combination, and to find the pair.
        The method of solving this can be much faster, as the time complexity 
        of this is  O(n ^ 2), but it can be O(nlogn).
    '''

    length = len(arr)
    for i in range(length):
        for b in range(i + 1, length):
            if arr[i] + arr[b] == target:
                return [arr[i], arr[b]]
    return []

# 2. Move Zeroes


def move_zeroes(arr: list) -> list:
    '''
        We first create a length reference and an index, which is initially 0.

        We then start a while loop, and on each iteration, we check if the element
        on index i in arr is 0. If it is, we pop it and append it to the end of the list.

        We decriment and then incriment the index because, it we didn't, 
        the next element after 0 would be skipped. We also need to decriment the length
        value on each modification, so that the zeros in  the end will not be iterated over.

    '''
    length = len(arr)
    i = 0
    while i < length:
        if arr[i] == 0:
            arr.append(arr.pop(i))
            length -= 1
            i -= 1
        i += 1
    return arr

# 3. Find three largest numbers

def find_three(arr: list) -> list:
    '''
        Initially we check if the length is less than 3.
        If it is, 0 is returned.

        Then I created 3 references to first 3 elements of the list, and just like
        how we find a single largest element in a list, we iterate over arr
        and just check all 3 numbers.

        if the element is greater than largest 1, we make largest1 equal to it and we basically push the previous 2 elements.
        This ensures that if we found a new largest number on one iteration, we preserve it during the next iterations.
        The equal or greater than ensures that we add duplicates if there are any.
    '''

    if len(arr) < 3:
        return 0
    
    largest1, largest2, largest3 = arr[0], arr[1], arr[2]
    for elem in arr:
        if elem >= largest1:
            largest3 = largest2
            largest2 = largest1
            largest1 = elem
            
        elif elem > largest2:
            largest3 = largest2
            largest2 = elem
        elif elem >= largest3:
            largest3 = elem
    return sorted([largest1, largest2, largest3])

# print(find_three([1, 5, 6, 12, 15, 18])) # 5, 1, 6: 6, 5, 1: 12, 6, 5: 15, 12, 6: 18, 15, 12

# 4. Best time to buy and sell stock

def stock(prices: list) -> int:
    '''
        Even though there is a better way to solve this,
        the current solution is O(n ^ 2).
    '''
    length = len(prices)
    profit = prices[1] - prices[0]
    answer = 1
    for i in range(length):
        for b in range(i + 1, length):
            if prices[b] - prices[i] > profit:
                profit = prices[b] - prices[i] 
                answer = b  
    return answer + 1

# print(stock([7, 1, 5, 3, 6, 4]))

# 5. Contains duplicate

def contains_duplicate(integers: list) -> bool:
    '''
        This can be done more easily by utilizing the count() method,
        but on low level, the solution can be written like this: 
        We instantiate a dictionary, iterate over the integers and if an integer is 
        in the dictionary ( as an element ), we increment it's value, else we set it as 1.

        Then we simply iterate over the values iterable and if a value is more than 1, that means that
        there are duplicates.

    '''
    occurences = dict()

    for elem in integers:
        if elem in occurences:
            occurences[elem] += 1
        else:
            occurences[elem] = 1

    for val in occurences.values():
        if val > 1:
            return True
        
    return False
    # There is a way shorter solution, and for that we use the any() and map() methods:
    # return any(map(lambda x: x > 1, [integers.count(i) for i in integers]))
    

# 6. Validate Subsequence Solved

def validate_sub(lst: list, seq: list) -> bool:
    '''
        We simply have an index starting with 0, and the length of the sequence. Considering that the elements may not be continous, 
        we must simply check that the order is correct. We iterate through lst and check if seq contains it. If it does, it must be equal
        to the next element for it to be a subsequence. If it isn't, we return False. Else the loop will break and we return True

    '''
    ind = 0
    sequence_length = len(seq)
    for elem in lst:
        if ind == sequence_length:
            break
        if elem in seq:
            if elem == seq[ind] and elem:
                ind += 1
            else:
                return False
    return True

# 7. Palindrome check

def is_palindrome(word: str) -> bool:
    '''
        I used slicing, therefore the solution is just a single line.

        The long way of solving this:  writing a for loop which would
        iterate over the elements in reverse, appending them to an empty list and then 
        calling ''.join(lst) on that list. If the final result is equal to the initial string,
        the word is a palindrome.
    '''
    return word == word[::-1]


# 8. Caesar Cipher Encryptor Solved
import string

def ceaser_encrypt(word: str, key: int) -> str:
    '''
        I use the string module for the solution, as i have easy access to lowercase ascii letters.
        
        1. First, we take the carry that we get after dividing the key with 26. If the passed key is 52 for instance,
        that doesn't change anything about the string. So, in order to get the actual key, we do this division.

        2. Then we start iterating over the letters in the word. We find the index of that letter in the alphabet, and then point to the
        position in a seperate variable. The reason for this is that, according to the problem, the letters must wrap around the alphabet.
        If one of the letters is y and we move by 2, it must wrap to a. Thats why i added the if statement.

        3. We append the finalized letter to the finalized_word list, which we pass to the join method to return the encrypted string.
    '''
    key = key % 26
    alphabet = string.ascii_lowercase
    finalized_word = []
    for i in word:
        ind = alphabet.index(i)
        position = ind + key
        if position >= 26:
            position = key - (26 - ind)
            print(position)
        finalized_word.append(alphabet[position])
    return ''.join(finalized_word)


# 9. First non-repeating character
def non_repeating(letters: list) -> int:
    '''
        The solution is similar to contains_duplicate, however, this time we 
        also add first indexes of elements to first_indexes.

        When we iterate over the occurences dictionary, 
        we check if the value of a given key is 1. If it is,
        we return the first_index of that element. Using a dictionary ensures that we get the 
        first element.
    '''

    occurences = dict()
    first_indexes = dict()

    for ind, letter in enumerate(letters):
        if letter not in first_indexes:
            first_indexes[letter] = ind

        if letter in occurences:
            occurences[letter] += 1
        else:
            occurences[letter] = 1
    print(first_indexes)
    for key in occurences:
        if occurences[key] == 1:
            return first_indexes[key]
        
    return -1

# 10. Valid Anagram

def valid_anagram(s: str, t: str) -> bool:
    '''
        Considering that when we are using sets the elements are
        not compared item by item, we can compare the strings easily.

        Removing duplicates isn't a problem here.
        As duplicates will be removed from both of the strings, the result
        will be the same.
    '''
    if set(t) == set(s):
        return True
    return False


# 11. Binary Search

def binary_search(nums: list, target: int) -> bool:
    '''
        The solution can be much faster, though this works.
        1. We first get the last index of the array ( and check if the element on that index is equal to the target or not )
        2. We create indexes dictionary so that when we do find the element, we know it's index in the initial state of the array.
        3. We start the while loop with the condition being != 0. When the modifid array will only have 1 element,
        the target will be compared to it before the next iteration, so it works.

        4. We do the binary search: we take the middle element, if that element is more than our target number,
        then we take the ,, left side " of the array, else we take the right side.
        This continues until either the middle element is our target, or we have a single element left in our array and
        it is the target. If it isn't, return -1.
    '''
    ind = len(nums) - 1
    if ind == 0 and nums[0] == target:
        return 0
    indexes = {elem : ind for ind, elem in enumerate(nums)}
    while ind != 0:
        ind = len(nums) // 2 
        if nums[ind] == target:
            return indexes[nums[ind]]
        if nums[ind] < target:
            nums = nums[ind::]
        else:
            nums = nums[:ind]
    return -1


# print(binary_search([1, 2, 3, 4, 5, 7, 8, 9, 10], 9))

# 1, 2, 3, 4, 5  ;   8


# 14. Validate BST, studied NeetCode for Depth first algorithm

def validate_bst(root, left_bound = float('-inf'), right_bound = float('inf')) -> bool:
    '''
        Essentially, we valiate the BST using boundaries. 
        For the head node, there are no boundaries, so we set -inf and -inf. 

        APPROACH:
        The approach is that, according to the properties of a BST, each left node must be less than
        the parent node, and each right node must be greater than the parent node. However, this statement alone
        isn't enough. If the head node is 5, with its right node being 7 and 7's left being 4, this condtion is satisfied,
        but it isn't a valid BST.

        So we also pass in a left boundary, set by the upper function calls.

        SOLUTION:

        1. If the root is None, then we got to the leaf node successfully and that side has been validated, so we return True.
        2. But if the value doesn't satify the boundaries, we return False.
        3. We then call the function recursively for both left and right nodes. If either one of them returns False, then the BST isn't valid.
    '''
    # def valid(node, left_bound, right_bound):
    if not root:
        return True
    if not (root.val < right_bound and root.val > left_bound):
        return False
    return validate_bst(root.left, left_bound, root.val) and validate_bst(root.right, root.val, right_bound)
    # return valid(root, float('-inf'), float('inf'))

# 19. Maximum Subarray
def max_subarr(nums: int) -> int:
    '''
        The approach is that we first create a cur_max to track the summary while iterating, and sub_max which we will use for the final
        answer we add each element to curr_max and if it is greater than sub_max, thats what we need.
        One case is that, considering that we are looking for the maximum sum of a subarray, we don't need negative sums. So if cur_sum
        is negative, we make it equal to 0, which basically implies that we are now summing up a new sub_array.
    '''
    length = len(nums)
    curr_max = 0
    sub_max = nums[0]
    for i in range(length):
        if curr_max < 0:
            curr_max = 0
        curr_max += nums[i]
        sub_max = max(curr_max, sub_max)
    return sub_max  

# print(max_subarr([1]))

# 20. House Robber

def rob(nums: list) -> int:
    '''
        Essentially, the problem is to find the largest sum of 
        non adjacent elements in an array.

        We partition the initial nums into sublists and calculate the maximum values of each one.
    '''
    hs1, hs2 = 0, 0
    for n in nums:
        temporary = max(n + hs1, hs2)
        hs1 = hs2
        hs2 = temporary
    return hs2



# 21. Minimum Waiting Time
def minimum_waiting_time(nums: list) -> int:
    '''
        1. We sort the array 
        2. As the execcution time is determined by the previous queries, we simply
        iterate over the elements and add the sum of the previous elements to the finalized answer.
        Sorting the array first ensures that shorter queries are executed first.
    '''
    queries_sorted = sorted(nums)
    answer = 0
    for i in range(len(queries_sorted)):
        answer += sum(queries_sorted[:i])
    return answer
# print(minimum_waiting_time([3, 2, 1, 2, 6]))

# 22. Class Photos Solved

def class_photo(reds: list, blues: list) -> bool:
    '''
        Considering that the first and second conditions are automatically met, we must check the third one.
        The simplest way to do this is to:
        1. Sort the lists 
        2. As they have the same length, we compare the elements on the same indexes. All elements from one of the lists
        should be higher than all of the second list's elements.
        3. We create 2 generator comprehensions and just call the any() method on them to check the condition stated above.
        The reason for 2 generators is that, considering on the input, both reds and blues can be the front/back row.
        So we just check both cases. 
    '''
    length = len(blues)
    sorted_reds, sorted_blues = sorted(reds), sorted(blues)
    iter1 = (sorted_reds[i] > sorted_blues[i] for i in range(length))
    iter2 = (sorted_reds[i] < sorted_blues[i] for i in range(length))
    return all(iter1) or all(iter2)

# print(class_photo([5, 8, 1, 3, 4], [6, 9, 2, 4, 5]))


# 1, 2, 3, 4, 5
# 6, 7, 8, 9, 10

# 23. Tandem Bicycle
def tandem_bicycle(reds: list, blues: list, fastest = True) -> int:
    '''
        Initially i wrote this like the variant i have commented. The logic is the same but there
        are a lot of redundant if statements, so i changed the implementation by sorting blues in reverse.
        Now, if we want to get the max result, we add the largest reds with smallest blues, and vice versa if we want the minimized speed. 
    '''

    length = len(reds)
    final_sum = 0
    # reds.sort()
    # blues.sort()
    # final_sum = 0
    # for _ in range(length):
    #     max_total = max(reds[0], blues[0])
    #     min_total = min(reds[-1], blues[-1])
    #     final_sum += max_total + min_total
    #     reds.pop(0)
    #     if not reds:
    #         break
    #     reds.pop(-1)
    #     blues.pop(0)
    #     blues.pop(-1)
    reds.sort()
    blues.sort(reverse = True)

    for i in range(length):
        if fastest:
            final_sum += max(reds[i], blues[i])
        else:
            final_sum += max(reds[i], blues[length - i - 1])

    return final_sum

# print(tandem_bicycle([5, 5, 3, 9, 2], [3, 6, 7, 2, 1]))


# 25. Task Assignment Solved, must refactor

def assign_tasks(k: int, tasks: list) -> list:
    '''
        The approach that I used is the following: 1, 3, 5, 3, 1, 4
        Considering that we must have k pairs, When the workers start doing the tasks, the amount of time it will take to finish
        the first ones will be the max value of each pair's 0th elements. This means that if the first elements will be minimized, so will
        the completion time for the first task.
        1, 1,  3, 3, 4, 5 
        So we sort the initial list and just append the min and max elements as a pair, then we pop both of them until k = 0.
    '''
    tasks.sort()
    final_pairs = list()
    # 1, 5  :   1, 4  :   3, 3 
    while k:
        max_val = max(tasks)
        min_val = min(tasks)
        final_pairs.append([min_val, max_val])
        tasks.pop(0)
        tasks.pop()
        k -= 1
    return final_pairs

print(assign_tasks(3, [1, 3,  5, 3, 1, 4]))
# 26. Parsing a boolean expression Solved, used a tutorial to grasp the problem



def evaluate(operators: list, expr: list):
    '''
        This function is called every time that we find a closing bracket ) while iterating over the expression.
        Considering that the latest expressions inside the expr list are the very expressions before the closing bracket,
        depending on the operator before these expressions, which is also the latest one in the operators list, we parse them.

        For instance, if the operator is AND ( & ), we set the initial answer as True ( so that it doesn't change the outcome ),
        and one at a time we pop the expressions and parse them. Once we get  to the opening bracket, we know that expression combination is 
        done, so we pop that bracket aswell.

        We use the same logic for all operators ( but we change the initial value of answer. In or it must be False to not change the outcome ).

        Finally we check the parsed expression combination. If it is True, we append 't', else 'f'.

    '''
    oper = operators.pop()
    ans = None

    if oper == '|':
        ans = False
        while expr[-1] != '(':
            ans |= (expr.pop() == 't')
        expr.pop()

    elif oper == '&':
        ans = True
        while expr[-1] != '(':
            ans &= (expr.pop() == 't')
        expr.pop()
    elif oper == '!':
        ans = not (expr.pop() == 't')
        expr.pop()

    if ans:
        expr.append('t')
    else:
        expr.append('f')

def parse_bool(expression: str) -> bool:
    '''
        The initial expression is passed to this function, and we call the evaluate() function every time we find a closing bracket.
        The actual lists are being updated here, according to the current character. 

        There is no need to return any of the updated lists. As lists are mutable and we pass the direct references, we can change their states
        directly in evaluate.

        After the parsing is done, there will be a single expression left in the expression_combs list, and we just return that bool.
    '''
    operators = []
    expression_combs = []
    length = len(expression)

    for i in range(length):
        current_char = expression[i]

        if current_char in ('&', '!', '|'):
            operators.append(current_char)
        elif current_char in ('t', 'f'):
            expression_combs.append(current_char)
        elif current_char == '(':
            expression_combs.append(current_char)
        elif current_char == ')':
            evaluate(operators, expression_combs)


    return expression_combs[-1] == 't'


# 28. FreqStack class Solved
class FreqStack:
    def __init__(self):
        self.freq_stack = list()
        self.counter = dict()

    def push(self, val: int) -> None:
        self.freq_stack.append(val)
        if val not in self.counter:
            self.counter[val] = 1
        else:
            self.counter[val] += 1        
    
    def pop(self) -> int:
        # print(self.freq_stack)
        max_freq = max(self.counter.values())
    
        for i in range(len(self.freq_stack) - 1, -1, -1):
            if self.counter[self.freq_stack[i]] == max_freq:
                element = self.freq_stack.pop(i)
                # print(element)
                self.counter[element] -= 1
                if self.counter[element] == 0:
                    self.counter.pop(element)
                return element

    def __str__(self):
        return str(self.freq_stack)

# 29. Marbles Solved

def marbles(weights: list, k: int) -> int:
    '''
        Essentially, we have to partition the given list into sublists, and the sum of the start and end indexes 
        of those sublists are what we need to compute the final result. 
        If we take a list: a, b, c, d, e, f
        
        We can partition it like this ( This is one case, but the outcome is the same for all cases ):
        a | b | c  d | e | f ... ( the line meaning the start of a new sublist ). The sum will be (a + a) + (b + c) + (d + e) + ...
        This means that one we group the elemets, both sides of the partitions will end up together.
        Meaning that to get all of the possible sums, we must iterate over the list and sum lst[i] and lst[i + 1]

        After we append all of the values to the sums list, we sort it, and considering that we have k bags, we iterate over it k - 1  times and
        , to get the max, sum each ith element ( as it is sorted in an ascending order ) and to get the min, we sum (len - i - 2)th element.

        Finally, we return max - min.

    '''
    sums = list()
    length = len(weights)
    min_sum = 0
    max_sum = 0
    for i in range(len(weights) - 1):
        sums.append(weights[i] + weights[i + 1] + weights[0] + weights[-1])
    sums.sort()
    for i in range(k - 1):
        min_sum += sums[i]
        max_sum += sums[length - 2 - i]
    return max_sum - min_sum



# 16. Nth fibonacci Solved

def nth_fibonacci(n: int) -> int:
    '''
        1. We create references to 0 and 1
        2. Considering that we already have 2 elements of the fibonacci sequence, 
        we iterate n - 2 times.
        3. On each iteration  i ( the first element ) will be changed to b, b will be changed to i + b.
        b will be the new element of the sequence, and it will be equal to (n-2) + (n-1)

    '''
    i, b = 0, 1
    for _ in range(n - 2):
        i, b = b, i + b
    return b

# 31. N pairs of parentheses Solved
def generate_parentheses(n: int) -> list:
    '''
        As the leetcode problem suggests, we use the backtracking recursive approach to solve
        this probem. If we keep track of the num of open and closed brackets, we can generate all solutions.
    '''
    solution = []
    answer = []
    def bcktrck(open:int, closed: int):
        '''
            1. Initially, we write th base case: if n number of parentheses are open and all n are closed,
            the length of the solution is of course 2 * n. If thats the case, we have a valid solution and we join the elements together and
            append the string to the answer list.

            2. We then chekc if n is greater than open. If this is the case, more brackets can be opened, so we 
            append the opening bracket to the solution and call the function again to track all of the possible combinations of that path.
            If that gets us to the valid solution, as i wrote in the first part, it gets added to the answers, if not we backtrack and 
            pop each bracket.

            3. if open is greater than closed then more brackets must be clased, so we use the same approach as the opening brackets and go down that path.

            4. This way, we get all of the valid combinations.
        '''
        if len(solution) == n * 2:
            answer.append(''.join(solution))
            return
        if n > open:
            solution.append('(')
            bcktrck(open + 1, closed)
            solution.pop()
        if open > closed:
            solution.append(')')
            bcktrck(open, closed + 1)
            solution.pop()
    # Here we call the function with the initial value of 0 for both open and closed, as we dont have any brackets inserted yet.
    bcktrck(open = 0, closed = 0)
    return answer


# 32. Unique Path Solved

def unique_path(m: int, n: int) -> int:
    '''
        In this case, n is the amount of columns. The approach is that from every position of the last row or the last column,
        there is only 1 way to get to the final position, being only going right or down, respectively.
        But what about, lets say, grid[m-1][n-1]? Well, considering that the robot can go either left or right, and from both left and right there
        is only 1 way to get to the final position, the amount of ways is 2. In the case of grid[m-1][n-2], as from down there is 1 way and from right
        there are 2, there are 3 ways and so on.

        Initially, we crate the last row, and it must have only 1s. Once we iterate, we also create a new row list with elements [1] * n.
        This is so that once we start going from right to left, we have the 1s both right and below initially, and we can keep adding the ways from
        right and down afterwards.

        The sum of all of the possible combinations will be the value of the first position, so we just return rows[0].
    '''
    rows = [1] * n

    for i in range(m - 1):
        current_row = [1] * n
        for b in range(n - 2, -1, -1):
            # We go from the right to left ( we start from n -2 to skip the rightmost position, which is 1 )
            current_row[b] = current_row[b + 1] + rows[b] # Value for each position will be equal to the value of down node + value of right node,
            # rows[b] being the down node and current_row[b] - right node.

        rows = current_row # We update the last row to the current one, so that we apply the same logic on the next iteration
    return rows[0] 

# 33. Minimum Flight cost

def min_cost(days: list, costs: list) -> int:
    '''
        We use backtracking to solve this problem. We i that we pass to it is the day, so that we can get the price
        based on the day inside the function.

        The base case is of course, if we increment i to the point where it gets out of bounds.

        We also have a cache where we store the kind of ticket we need for the ith day. If we get to that day again, 
        we don't have to compute the value again.

        If it is not in the cache however, we zip the costs and the amount of days those tickets cover, and we start a while loop 
        with the condition bein that days[b] ( so the days after the current day ) will be less than the current day + duration of the
        ticket. We increment it each time and then call the function recursively to use the same logic on
        the next days.

        We cache thie results and just return that inside the inner function, whereas in the outer function's scope we just call
        the recursivel bcktrck function starting from day 0.
    '''
    cache = {}
    def bcktrck(i):
        if i == len(days):
            return 0
        
        if i in cache:
            return cache[i]
        cache[i] = float('inf')
        b = i
        for price, duration in zip(costs, [1, 7, 30]):
            while b < len(days) and days[b] < days[i] + duration:
                b += 1
            cache[i] = min(cache[i], price + bcktrck(b))
        return cache[i]
    return bcktrck(0)

# 37. Partition Labels, Solved

def partition_labels(s: str) -> list:
    '''
       1. We create first and last references so that we can append the partitions later on.
       2. Then we create a dicitonary where we store the last occurences of the characters, in order.
       3. We start iterating over the string once again, we determine the last occurence of the current character
       using the max() method and then we compare. If i is the last index of the occurence, that means that
       we found one partition, and we append that to legnths and increment first so that we start the next partition.

       Finally, we return lengths.
    '''
    lengths = []
    first = 0
    last = 0
    last_occurrences = {ch: i for i, ch in enumerate(s)}
    for i, ch in enumerate(s):
        last = max(last, last_occurrences[ch])
        if i == last:
            lengths.append(i + 1 - first)
            first = i + 1
    return lengths

# print(partition_labels("ababcbacadefegdehijhklij"))
# "ababcbacadefegdehijhklij"
# 17. Minimum coins for change 

def min_coins(n: int, denoms: list) -> int:
    '''

    '''
    dyn = [n + 1] * (n + 1)
    dyn[0] = 0

    for a in range(1, n + 1):
        for c in denoms:
            if a - c >= 0:
                dyn[a] = min(dyn[a], 1 + dyn[a - c])
    return dyn[n] if dyn[n] != n + 1 else -1


# print(min_coins(n = 11, denoms = [11, 2, 5]))



class TreeNode:
    '''
        The tree will be built by using the instances of this class.
        The head node will also be a TreeNode instance.
    '''
    def __init__(self, value):
        self.value = value
        self.right = None
        self.left = None

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = TreeNode(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = TreeNode(value)
            else:
                self.right.insert(value)

    # 12. Binary Search Tree Solved
    def find_closest(self, target: int, result: int = 0, difference: int = 0) -> int:
        diff = abs(target - self.value)
        if not difference:
            difference = diff
            result = self.value

        if diff < difference and target != self.value:
            difference = diff
            result = self.value
            # print(result, self.value)

        left_side = None
        right_side = None
        if self.right:
            right_side = self.right.find_closest(target, result, difference)
            # print(right_side)
        if self.left:
            left_side = self.left.find_closest(target, result, difference)
            # print(left_side)

        if left_side and right_side:
            # print(left_side, right_side)
            if abs(target - right_side) < abs(target - left_side):
                return right_side
            else:
                return left_side
        elif right_side:
            return right_side
        elif left_side:
            return left_side
        else:
            # print(result)
            return result

    # 13. BST Traverse Solved
    def postorder_traverse(self, arr: list = [], first_call = True) -> list:
        '''
            We append the value of the current node only when there are more child nodes
            to traverse. Hence, according to the test, 
            the first output will be the last left node 1, followed by 2 and 5 ( the last 5 node ) and 5. The parent
            node will be the last one to be printed, considering that it gets ,,exhausted" the last.
        '''
        if self.left:
            self.left.postorder_traverse(arr, False)
        if self.right:
            self.right.postorder_traverse(arr, False)
        arr.append(self.value)
        return arr
        
    def inorder_traverse(self, arr: list = [], first_call = True) -> list:
        '''
            A node's value gets appended once all of the left nodes are exhausted. 
        '''
        if self.left:
            self.left.inorder_traverse(arr, False)
        arr.append(self.value)
        if self.right:
            self.right.inorder_traverse(arr, False)
        return arr
    
    def preorder_traverse(self, arr: list = [], first_call = True) -> list:
        '''
            A node's value gets appended once it is visited. Meaning that the outcome of this is exactly how the
            tree is traversed. the first node will be the parent node, then all of the left nodes, respectively. Then right and etc..
        '''
        arr.append(self.value)
        if self.left:
            self.left.preorder_traverse(arr, False)
        if self.right:
            self.right.preorder_traverse(arr, False)
        return arr
        
    
    def max_depth_cls(self, depth: int = 1) -> int:
        '''
            The initial solution that i wrote which works on the actual
            node instances.
        '''
        left_depth = None
        right_depth = None
        if self.left:
            left_depth = self.left.max_depth(depth + 1)
        if self.right:
            right_depth = self.right.max_depth(depth + 1)
        if not self.left and not self.right:
            return depth
        if left_depth and right_depth:
            return max(right_depth, left_depth)
        elif right_depth:
            return right_depth
        elif left_depth:
            return left_depth


# 15. Maximum Depth (Solved, must practice more)
def max_depth(root) -> int:
    '''
        Though I wrote the class version, as we pass the root seperately, this is the solution.
        We basically traverse the tree and if we get to the last leaf nodes, it returns 0. In other cases, it increments the count for
        both left side and the right side. 
    '''
    if not root:
        return 0
    left_sums = max_depth(root.left)
    right_sums = max_depth(root.right)
    return max(right_sums, left_sums) + 1




class ListNode:
    def __init__(self, val = 0, next = None):
        self.next = next
        self.val = val


# 24. Valid Starting City (Must test, works for the given test case)

def valid_start(distances: list, fuel: list, mpg: int) -> int:
    '''
        To check the result for a full circle, we have to iterate over the
        distances list twice. 

        We create a starting index of 0, on each iteration we track the current fuel for the car, which is incremented by
        miles per gallon * fuel on each iteration. If current fuel is enough to get to the next city, then that index is a potential
        valid city, and we decrement it by the distance that it takes to get to the next city. If not, we set the total fuel to 0 to
        start tracking again. We do this twice, and as we increment the start_index on each valid condition, we return it to get the first
        valid city.
    '''
    valid_route = list()
    can_travel = 0
    i = 0
    second_iter = False
    start_index = 0
    length = len(distances)
    while i < length:
        can_travel += fuel[i] * mpg
        print(valid_route, can_travel)
        if can_travel >= distances[i]:
            can_travel -= distances[i]
            valid_route.append(i)
        else:
            can_travel = 0
            start_index = i + 1
            valid_route.clear()
        if i == length - 1:
            if not second_iter:
                i = 0
                second_iter = True
                continue
            else: break
        i += 1
    return start_index




# print(valid_start([10, 20, 30, 10], [1, 1, 1, 1], 10))

# 34. Swap Pairs Solved 

def swapPairs(self, head: ListNode):
    '''
        The name and the arguments for the function are taken from the leetcode page.
        I am using a dummy node to simplify the logic, as swapping the node whilst it being a head node
        will make things more complicated.

        Essentially, all we need to do is to swap the references. As the second node will now be the first node,
        we are setting scnd.next to current, and as the first node will now be the second node, it must point to the first node of the
        next pair, so we set current.next to the next_pointer. 

        Considering that we are also updating back, which was the dummy pointer at first, after every iteration we set back to 
        current, so that this logic can repeat for every pair.

        The conditions themseles ( current and current.next ) just check that there is a pair of nodes left until we do these operations.
    '''
    initial_dummy = ListNode()
    back, current = initial_dummy, head
    while current and current.next:
        next_pointer = current.next.next
        scnd = current.next
        scnd.next = current
        current.next = next_pointer
        back.next = scnd
        back = current
        current = next_pointer

    return initial_dummy.next

# 35. Linked List Integers

def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    '''
        The initial algorith that i wrote and  commented has several problems. 
        The memory was not being utilized correctly, as i traversed over the linked lists 
        seperately, instantiated lists, strings, creating a new one for every digit added, and so on.

        As i found out the solution can be written inside a single while loop, adding the numbers digit by digit,
        instead of reconstructing them as integers.

        Now, we create a dummy node and a pointer to it initially, including a carry integer, used for adding the digits one at a time.

    '''
    # traverse_l1 = l1
    # traverse_l2 = l2
    
    # num_1 = []
    # num_2 = []
    # finalized_num1 = ''
    # finalized_num2 = ''
    # while traverse_l1:
    #     num_1.append(traverse_l1.val)
    #     traverse_l1 = traverse_l1.next
    # while traverse_l2:
    #     num_2.append(traverse_l2.val)
    #     traverse_l2 = traverse_l2.next
    # for i in num_1[::-1]:
    #     finalized_num1 += str(i)

    # for i in num_2[::-1]:
    #     finalized_num2 += str(i)
    # finalized_number = int(finalized_num1) + int(finalized_num2)
    # final_number = ListNode(finalized_number % 10)
    # finalized_number = finalized_number % 10
    # next_number = finalized_number
    # while finalized_number:
    #     next_number.next = ListNode(finalized_number % 10)
    #     next_number = next_number.next
    #     finalized_number = finalized_number % 10
    # return final_number
    dummy = ListNode()
    curr = dummy
    carry = 0

    while l1 or l2 or carry:
        val_1 = l1.val if l1 else 0
        val_2 = l2.val if l2 else 0
        total = val_1 + val_2 + carry
        carry = total // 10

        curr.next = ListNode(total % 10)
        curr = curr.next

        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    return dummy.next

# 36. Two Anagram Strings Solved

def min_steps(s: str, t: str) -> int:
    '''
        We count the amount of each character in both s and t,
        and we iterate over s_counts. If an element in s occurs more than it does in t, that means that
        s[elem] ( count ) - t[elem] must be the amount of characters to be replaced.

        Therefore, we add that to final_amount. If a character in s isn't in t_counts, then we add the whole count.
        Finally, we return final_ammount
    '''
    s_counts = {x: s.count(x) for x in set(s)}
    t_counts = {x: t.count(x) for x in set(t)}
    print(s_counts, "\n", t_counts)
    final_ammount = 0
    for letter in s_counts:
        if letter not in t_counts:
            final_ammount += s_counts[letter]
        elif s_counts[letter] > t_counts[letter]:
            final_ammount += s_counts[letter] - t_counts[letter]
        else:
            pass

    return final_ammount



# 39. Array Of Queries Solved

def occurencesOfElements(nums: list, queries: list, x: int) -> list:
    '''
        We create an indexes dictionary, where the keys will be elements and the values will be
        lists which contain the indexes of those elements, in order. So on the 0th index there will be the index of the first occurence,
        on 1th the second occurence and so on. 

        Afterwards, we iterate through queries, and on each element we try to append indexes[x][i - 1]. If i is 3, it means that we want the 
        index of the third occurence, which will be on index 2 of indexes[x]. To avoid comparing i and  the length of indexes[x],
        i just wrote the logic inside a try except block. In the case of an exception, there is no ith occurence, so we append -1.

        FInally, we return positions.
    '''
    indexes = dict()
    positions = []
    for i in range(len(nums)):
        if nums[i] not in indexes:
            indexes[nums[i]] = [i]
        else:
            indexes[nums[i]].append(i)
    for i in queries:
        try:
            positions.append(indexes[x][i - 1])
        except:
            positions.append(-1)
    return positions    


# print(occurencesOfElements([1,4,3,3,6,4,8,3,10], [1,2,1,1,1,1,2,2,1,1], 7))

# 38. Count Substrings Solved
def count_substrings(s: str) -> int:
    '''
        To solve this, we are using the sliding window approach.
        1. We create a count dictionary. We use a defaultdictionary to provide factory values.

        2. Essentially, for every substring between two references of left and end, if that given substring
        is valid, then all of the other substrings ( last elements being characters after the end index ) are also valid.
        So if we have 1 substring and there are 4 other characters after end, then there are 5 total substrings for this combination.

        3. We create the end (right) pointer in the for loop, and we add the count for that char inside our counts dictionary.
        We then write a while loop inside. The condition is for the length of the counts dict being 3, as that implies that
        a b and c are occuring at least once in the current sequence. 

        4. Inside the loop, the result incrementation line just adds the ammount of chars left + 1 to the result,
        and then slides the left window right by one, decrementing the count of it in the process.
        If the substring that is left is still valid, the same logic is repeated, but if the count was 1 and now we moved the pointer right,
        ( so if the count is now 0 ), we pop that item, and the loop breaks. 
        
        5. Then the right pointer is moved one element to the right and the same logic is repeated.
    '''
    left = 0
    result = 0
    char_counts = defaultdict(int)
    for end in range(len(s)):
        char_counts[s[end]] += 1
        while len(char_counts) == 3:
            result += len(s) - end
            char_counts[s[left]] -= 1
            if char_counts[s[left]] == 0:
                char_counts.pop(s[left])
            l += 1

# 40. Airplane Seat Assignment 

def airplane_seat(n: int) -> float:
    if n == 1:
        return 1
    elif n >= 2:
        return 0.5

# 41. Palindrome Partitioning Solved

def palindrome_partition(s: str) -> list:
    partitions = []
    partition = []
    # length = len(s)
    # def bcktrck(a: str, first: int = 0, last: int = 0):
    #     opp = a[first : last]
    #     if first == length:
    #         return
    #     if opp and opp == opp[::-1]:
    #         partitions.append(opp)

    #     if last < length:
    #         bcktrck(a, first, last + 1)
    #     else:
    #         bcktrck(a, first + 1, first + 1)
    # bcktrck(s)
    # return partitions
    def bcktrck(i):
        '''
            The initial solution didn't work as excpected, so now it is written using a for loop to
            backtrack.
            We start iterating over the characters in the string and check every possible substring starting from a given
            index. 
            
            If that substring is a palindrome, then we append it to the partition and call the function recursively,
            passing b + 1 to the function, so that it now checks the next index. Then, to backtrack ( so once we checked 
            all of the substrings for the initial partition list), we pop the elements and then check the next element
            and so on.
        '''
        if i >= len(s):
            partitions.append(partition.copy()) # Considering that we keep modifying the list object that
            # partition references, we pass a copy instead of the actual reference.
            return
        for b in range(i, len(s)):
            potential = s[i : b + 1]
            if potential == potential[::-1]:
                partition.append(potential)
                bcktrck(b + 1)
                partition.pop()
    bcktrck(0)
    return partitions
# print(palindrome_partition('aab'))

# 42. Rotated Digits
def rotated_digits(n: int) -> int:
    '''
        According to the problem, the new integer must be different from
        the previous one to be valid. For this to be true, it must contain at least
        one of the digits which change after rotating by a 180 degrees.

        We create a list of elements using the range iterator and just check if 
        a number contains one of those digits. if it does, it is valid and we increment the count integer.
        
        Finally, we return count.
    '''
    numbers = list(range(1, n))
    count = 0
    for num in numbers:
        num_str = str(num)
        if '3' in num_str or '4' in num_str or '7' in num_str:
                continue
        elif '2' in num_str or '9' in num_str or '5' in num_str or '6' in num_str:
            count += 1
    return count

# 43. Extra characters in a string
def min_extra_char(s: str, words: dict) -> int:
    '''
        The solution utilizes recursion.
        On each call, we check whether or not the substring from i to j + 1 is in words.
        If it is, we simply calculate the result and cache it, so that on future recursive calls,
        we don't calculate it all over again.
    '''
    cache = dict()
    def char(i):
        if i == len(s):
            return
        if i in cache:
            return cache[i]
        res = 1 + char(i + 1)  

        for j in range(i, len(s)):
            if s[i : j + 1] in words:
                res = min(res, char(j + 1))
        cache[i] = res
        return res
    return char(0)        

from functools import lru_cache

# 44. Integer Break, Must work on it
def integer_break(n: int) -> int:
    '''
        This is a dynamic programming approach, meaning that we break the initial problem down.
        The actual dp function's base case is that if i == 0, if we pass an integer which can not be broken down 
        anymore, we return 1.

        Else, we set a max_product to the lowest number so that we can calculate it later on. 
        The range is determined by the number which is passed to the function. If it isn't the initial number,
        we set the final_range to i + 1, else, considering that we cant go above, we set it to i.


        Then we start a for loop and on each iteration we determine the maximum between product's past value and 
        and recursive call multiplied by the current value in the loop. Essentially, in each recursive call, the max prdct will be determined
        and we get the finalized value.

        To initiate the recursion, we call dp with i = n as it's initial value.
    '''

    @lru_cache
    def dp(i):
        if i == 0:
            return 1
        max_prdct = float('-inf')
        final_range = i + 1 if i != n else i
        for j in range(1, final_range):
            max_prdct = max(max_prdct, dp(i - j) * j)
        return max_prdct
    return dp(n)


def binary_search_custom(starts: list, target: int) -> int:
    '''
        Considering that the binary search that i wrote was more complex than this,
        using the same implementation made this more difficult. So this function does the binary search
        in a more simple way. Instead of using indexes dictionary and constantly modifying the passed list ( which would maake this approach impossible ),
        we simply have 2 pointers which we move accordingly.

        If the element is greater than target, thats what we need, so we change the indexes.
        Finally, if that element was found, we return starts[left][1] ( The value itself is the second element ), if not, we return -1.

    '''
    left = 0 
    right = len(starts) - 1
    while left <= right:
        mid = (left + right) // 2
        if starts[mid][0] >= target:  
            right = mid - 1  
        else:
            left = mid + 1  
    if left < len(starts):
        return starts[left][1]      
    return -1  

# 46. Valid Interval
def valid_interval(intervals: list) -> list:
    '''
        First we sort the indices, so that the binary search is more efficient and actually finds the minimum value.
        Then we instantiate final_indicies, which will store the found indexes.

        Then we start iterating over the initial list intervals. On each 
        iteartion, we call the custom binary search function and append the result. 
        Finally, we just return final_indices.
    '''
    start_indices = sorted((start, i) for i, (start, end) in enumerate(intervals))
    final_indices = list()
    for start_elem, end_elem in intervals:
        nearest_ind = binary_search_custom(start_indices, end_elem)
        final_indices.append(nearest_ind)
    return final_indices


# 47. Valid Stack Solved
def valid_stack(pushed: list, popped: list) -> bool:
    '''
        For the passed operations to be a valid stack sequence, they must occur in order.
        So for instance, if the first element of popped is 4, it can only be done if the last operation was pushing 4.

        This means that we have to iterate over the pushed list, and each time we find an element which is equal to popped[ind_popped]
        (ind_popped is 0 initially, as the first operation can not be skipped ), we have to pop both elements from both lists,
        as that is the only valid way the sequences can execute.
    '''
    ind_pushed = 0
    ind_popped = 0
    
    while pushed and ind_pushed < len(pushed):
        # print(pushed, popped, ind_pushed)
        first = False
        second = False
        if pushed[ind_pushed] == popped[ind_popped]:
            # print(pushed[ind_pushed])
            pushed.pop(ind_pushed)
            popped.pop(ind_popped)
            if ind_pushed:
                ind_pushed -= 1
            first = True

        if ind_pushed < len(pushed) - 1:
            print(ind_pushed, pushed)
            if pushed[ind_pushed + 1] == popped[ind_popped]:
                pushed.pop(ind_pushed + 1)
                popped.pop(ind_popped)
                second = True
            # ind_popped += 1
        if not first and not second:
            ind_pushed += 1
        
    if pushed:
        return False
    return True
            
    

# 48. Temperatures Solved

def manage_temperature(temp_pair: tuple, temp_stck: list, indexes: list):
    '''
        There are multiple ways to do this, though the best approach is
        to use a stack (or a list in our case), which we use to track the elements and the indexes, so that we can compare
        elements more easily.

        This function will be called for each element in temperatures.
        The temp_stck is where the temps are appended, so we start iterating over it ( in reverse of course, to mimic a stack ) 
        and check if theat element is less than the current one. If it is, we found the nearest warmer temperature, so we
        pop that pair from temp_stck, decriment i and set the indexes element on at that popped element's index to the difference
        between the warmer one and the old one, as that is the distance.

        If an element in temp_stck is not less, that means that element was the nearest one to the temperatures before that, so
        there is no need to keep iterating and we just break to avoid redundant iterations.

    '''

    # while temp_stck and temps[temp_stck[-1]] <  temps[temp_ind]:   
    #     less_ind = temp_stck.pop()
    #     indexes[less_ind] = temp_ind - less_ind
    if temp_stck:
        i = len(temp_stck) - 1
        while i >= 0:
            if temp_stck[i][1] < temp_pair[1]:
                indexes[temp_stck[i][0]] = temp_pair[0] - temp_stck[i][0]
                temp_stck.pop(i)
                i -= 1
            else:
                break


def temps(temperatures: list) -> list:
    '''
        Initially, we instantiate final_indexes list where we will store the differecne between indexes.
        temp_stck will immitate a stack and will be the object modified in manage_temperature function.
    '''
    final_indexes = [0] * len(temperatures)
    temp_stck = list()

    for ind, elem in enumerate(temperatures):
        
        manage_temperature((ind, elem), temp_stck, final_indexes)
        temp_stck.append((ind, elem)) # Once we modified temp_stck and popped the previous elements, we append the new one.
    return final_indexes
        

# print(temps([73,74,75,71,69,72,76,73]))
# 18. Climbing Stairs Studied neetcode, learning Dynamic Programming
def climb_stairs(n: int) -> int:
    '''
        The dynamic programming approach solution for this problem is identical to the fibonacci sequence,
        as the logic itself is simple once we break down the logic into smaller pieces.
        If there are 5 stairs for instance, if we start from the last step, as 5 is the base case, there is 1 way.
        There is also 1 way to get from step 4 to 5.
        However, from 3, we now have 2 ways to get to 5 ( 1 step at a time or 2 steps )
        From 2, considering that we have 3 combinations in total for step 3 and 4, we have 3 total ways.
        And etc, when we get to 0, we just sum up the combinations for stairs 1 and 2. Because of this, the logic is the exact same as
        a fibonacci sequence. 
    '''
    i, b = 0, 1
    for _ in range(n):
        i, b = b, i + b
    return b

# 27. Sliding Puzzle Solved
def two_by_three(board: list):
    '''
        To make this problem easier, the 2x3 matrix is represented as a string initially. We also store all of the
        possible moves for each node inside a list, and we have a list to track the states of the board.

        The list that we have for the states also contain the amount of moves done to get to that state, which is calculated inside the for loop.
        Each time 0 is on index i-th, we check all of the combinations of moves it can make from that posiiton. We swap places with the
        possible position and check if that position is calculated or not. If it isn't, we add it to the set and append a new element to the 
        state list, which resembles a queue in our case.

        We repeat this for each state inside the queue until it is empty. If a valid board is constructed, it returns the moves instantly.
        If not,  
    '''
    initial_state = ''.join(str(c) for row in board for c in row)
    state_set = set(list(initial_state))
    target = '123450'
    possible_moves = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4],
        4: [3, 1, 5],
        5: [2, 4]
    }

    q = [(initial_state, 0)]

    while q:
        current, moves = q.pop(0)
        if current == target:
            return moves
        
        indx = current.index('0')

        for possible in possible_moves[indx]:
            new_state = list(current)
            new_state[indx], new_state[possible] = new_state[possible], new_state[indx]
            new_state = ''.join(new_state)
            if new_state not in state_set:
                state_set.add(new_state)
                q.append((new_state, moves + 1))
    return -1


# 50. Different ways to group Solved

def different_groups(expr: str) -> list:
    '''
        For every expression, we must split it once we see a numerical operator. For instance, for 2 - 1  - 1, we can split it 
        at the first - and we will have: 2 and (1 - 1), or the second minus and we will have (2 - 1) and 1.

        Essentially, once we break it into 2 parts, we must then do the same for the left and right sides recursively.
        The base case will be when the expression contains a single number, as we can not break that expression down any further.

        If it isnt, we get the left_results and the right_results for each function call, and we parse those results according to the
        numerical operation that is in the expression.
        Finally, we return the res list

    '''
   

    res = []
    for i in range(len(expr)):
        if expr[i] in '-+*':
            left = expr[:i]
            right = expr[i + 1:]

            left_rslts = different_groups(left)
            right_rslts = different_groups(right)

            for b in left_rslts:
                for r in right_rslts:
                    if expr[i] == '+':
                        res.append(b + r)
                    elif expr[i] == '-':
                        res.append(b - r)
                    elif expr[i] == '*':
                        res.append(b * r)
    if expr.isdigit():
        return [int(expr)]
    return res

# 45. Perfect square Solved

def perfect_squares(n: int) -> int:
    '''
        The solution to this problem is also written using a dynamic progrmaming approach. 
        First we instantiate a cache list. All of the values in cache are n because, in worst case scenario, we will need
        n amount of 1-s to get to that target.

        Then we start breaking the problem into subproblems. For each target, we are going to find the minimum number of
        numbers before that target that are perfect squares and sum up to that target.

        If the square is greater than target, we don't need that value so we break the inner loop. Else, we cache the minimum between 
        the previous value and cache[target - square] + 1. 

        Finally, we return the last element, which will contain least amount. 
    '''
    cache = [n] * (n + 1)
    cache[0] = 0

    for target in range(1, n + 1):
        for s in range(1, target + 1):
            square = s ** 2
            if target < square:
                break
            cache[target] = min(cache[target], cache[target - square ] + 1)
    return cache[n]