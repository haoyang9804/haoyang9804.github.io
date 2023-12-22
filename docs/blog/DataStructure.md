# A Summary of Selecting Suitable Data Structures to Resolve Abstract Problems.

## Up and Down (or Down and Up) Sequence

Given a sequence of non-decreasing integers of length n, performs the following step for n times.

+ add a new integer *I* at the end and change all integers's values to *I*'s value from the end the the head until meeting an integer that is not larger than *I*.

> Q: How to make it O(n)?

> A: Instead of maintaining a deque of intergers, we can maintain a pair (integer, the count of its occurrences). If the inserted value is small, the sequence will collapse into a much shorter one ([(1, 1), (2, 1), (3, 1), (4, 1), (4, 1), (4, 1), 3] -> [(1, 1), (2, 1), (3, 5)], where the last 3 is the inserted integer which does not form a pair then). 

https://codeforces.com/contest/1905/problem/D