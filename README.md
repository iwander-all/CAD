Programming Lab 1 – curves 
	
Using any Computer Graphics programming language such as MATLAB, write a simple program that has the following functions:

(1)	The user can specify the coordinates of a list of 3D points, using BOTH the following ways: 

	(a) Keyboard input; AND 
	(b) Read from a text file (see the Note below).
	
(2)	The program constructs the Bezier curve using the points got from (1) as the control polygon and displays it on the screen.

(3)	The program should be able to save the control points of the curve into a text file which can then be input again into the program (i.e., the (b) of (1)).

Note: eventually for the other projects you will be asked to define a Bezier curve input from a text file (i.e., (b) of (1)), so, it is suggested you do it now; the format of the text file must be this:

Line 1		m 		/* the number of Bezier curves */

Line 2		n1 		/* the number of control points of the first Bezier curve */

Line 3		Either 0 or 1  	/*0 means the curve is closed, 1 means it is open */

Line 4		the XYZ coordinates of the 1st control point   /*separated by space or “,” */

Line 5		the XYZ coordinates of the 2nd control point 

.
.
.

Line n1+3	the XYZ coordinates of the n1th control point		

Repeat for the next m-1 Bezier curves.

Example:
2
4
1
1.5		0.4	-3

1.7   	-2.56   	4.78
0   		245.6 	10
0		0	0
3
0
3		5	-6
-4		3.5	7
1.5		-5	-10.5
