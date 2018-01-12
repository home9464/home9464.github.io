# Problem
We define the problem as a multinormial classification. 
where each label has two or more different features. 

To simplify the problem, let us assume we have a dataset like this:

|label   | feature1  | feature2  |
|---|---|---|---|---|
| A1  | 'Brown fox jumps on fox'  | '0x123, 0x234, 0x345, 0x123' |
| A2  | 'Brown fox jumps on fox'  | '0x123, 0x234, 0x345, 0x123' |
| A3  | 'Brown fox jumps on fox'  | '0x123, 0x234, 0x345, 0x123' |
| A4  | 'Brown fox jumps on fox'  | '0x123, 0x234, 0x345, 0x123' |
| A5  | 'Brown fox jumps on fox'  | '0x123, 0x234, 0x345, 0x123' |
| A6  | 'Brown fox jumps on fox'  | '0x123, 0x234, 0x345, 0x123' |
| ...  | ...  | ...  |
| A100  | 'Brown fox jumps on fox'  | '0x123, 0x234, 0x345, 0x123' |

