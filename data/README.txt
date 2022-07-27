CVC-ClinicDB

--------------------------------------------------
Introduction
--------------------------------------------------

CVC-ClinicDB is a database of frames extracted from colonoscopy videos. 
These frames contain several examples of polyps. In addition to the frames, we provide the ground truth for the polyps. 
This ground truth consists of a mask corresponding to the region covered by the polyp in the image.

CVC-ClinicDB has been generated from 25 different video studies. For each study at least a sequence containing a polyp was extracted. Considering this, CVC-ClinicDB database is composed by frames from 29 different sequences containing a polyp. 

Finally, a set of frames were selected from each sequence, paying particular attention in showing several points of view of the polyp. 

--------------------------------------------------
Description
--------------------------------------------------

The database consists of two different types of images:
1) Original images: original/frame_number.tiff
2) Polyp mask: ground truth/frame_number.tiff

The correspondence between the number of frame and the video sequence is as follows: 

       ----------------------------------
       |                  |		           |
       |   Frame Number   |  Sequence	      |
       |                  |		           |
       ----------------------------------
       |		          |	|
       |      1 to  25    |	 1	|
       |     26 to  50    |	 2   |
       |     51 to  67    |	 3 	|
       |     68 to  78    |	 4 	|
       |     79 to 103    |	 5 	|
       |    104 to 126    |	 6 	|
       |    127 to 151    | 7	|
       |    152 to 177    |	 8	|
       |    178 to 199    |	 9	|
       |    200 to 205    |	10	|
       |    206 to 227    |	11	|
       |    228 to 252    |	12	|
       |    253 to 277    |	13   |
       |    278 to 297    |	14 	|
       |    298 to 317    |	15 	|
       |    318 to 342    |16 	|
       |    343 to 363    |	17 	|
       |    364 to 383    |18	|
       |    384 to 408    |19	|
       |    409 to 428    |20	|
       |    429 to 447    |	21	|
       |    448 to 466    |	22	|
       |    467 to 478    |23	|
       |    479 to 503    |24	|
       |    504 to 528    |	25	|
       |    529 to 546    |	26	|
	  |    547 to 571    |27	|
       |    572 to 591    |28	|
       |    592 to 612    |	29	|
       ----------------------------------


CVC-ClinicDB is the database to be used in the training stages of ISBI 2015 Challenge on Automatic Polyp Detection Challenge in Colonoscopy Videos. 

--------------------------------------------------
Copyright
--------------------------------------------------

Images from folder 'Original' are property of Hospital Clinic, Barcelona, Spain.

Images from folder 'Ground Truth' are propery of Computer Vision Center, Barcelona, Spain.

--------------------------------------------------
Referencing
--------------------------------------------------

The use of this database is completely restricted for research and educational purposes. The use of this database is forbidden for commercial purposes.

If you use this database for your experiments please include the following reference:

Bernal, J., Sánchez, F. J., Fernández-Espárrach. G and Rodríguez, C. 'CVC-ClinicDB' (related publication to be available soon)