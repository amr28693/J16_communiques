# MOTHER script is the GENERATOR of PERSISTENCE FILES

Then, the CHILDREN scripts, MASTER and RECEIVER can be run on the MOTHER--generated PERSISTENCE FILES.

# The REPO begins with a MOTHER session 0 PERSISTENCE FILES generated and copied to the respective CHILDREN folders.

To begin the experiment: 
* Download the repository. 
* Open either the MASTER or RECEIVER subfolder
* If MASTER, run according to MASTER protocol
* If RECIEVER, run according to RECEIVER protocol
* The Session 0 MOTHER--generated PERSISTENCE FILES will now be OVERWRITTEN by the new CHILD-GENERATED data
* Take your CHILD--generated session data and compare to the opposite SIBLING's data (so, if MASTER, compare to RECEIVER, and if RECEIVER, compare to MASTER--generated data).
* Load combined data into ANALYSIS subfolder and run the analysis script


## To send a message:

Both systems running, torus playing, coupled
Q3 positive = "0"
You perturb Master with magnet, Q3 flips negative = "1"
Remove magnet, Q3 returns positive = "0"
Repeat for each bit

## To receive:
You don't see it in real-time right now. You run the experiment, stop both systems, pull the prosecutor logs, run message_tx.py, and it shows you the Q3 trace and decodes the bits.
This run will tell you:

Do the new scripts couple like the old ones did?
Can you intentionally flip Q3 with the magnet?
Does the flip pattern show up in the decoder?

Start simple. Run for a few minutes stable, then do one clear magnet perturbation, wait, do another. Look for two flips in the trace.

### The magnet is your telegraph key.

# ASCII "A" = 01000001
Hold magnet OFF  (Q3 positive) = 0  [10 sec]
Hold magnet ON   (Q3 negative) = 1  [10 sec]
Hold magnet OFF  (Q3 positive) = 0  [10 sec]
Hold magnet OFF  (Q3 positive) = 0  [10 sec]
Hold magnet OFF  (Q3 positive) = 0  [10 sec]
Hold magnet OFF  (Q3 positive) = 0  [10 sec]
Hold magnet OFF  (Q3 positive) = 0  [10 sec]
Hold magnet ON   (Q3 negative) = 1  [10 sec]
That's 80 seconds to send one letter.
message_tx.py already has the ASCII decoder built in. It looks at the collapsed bit string and tries to convert every 8 bits to a character.

"h" = 01101000
"i" = 01101001
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]

0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
160 seconds total. Just under 3 minutes.

"y" = 01111001
"o" = 01101111
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]

0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
160 seconds.


"f" = 01100110
"a" = 01100001
"r" = 01110010
"t" = 01110100
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]

0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]

0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]

0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
1 - magnet ON   [10 sec]
0 - magnet OFF  [10 sec]
0 - magnet OFF  [10 sec]
320 seconds. Just over 5 minutes.
