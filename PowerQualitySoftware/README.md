What is this Project?






Process Steps:

1. setting up dataloader.py
   The purpose of this file is to read and access the values of the input csv file, that will contain the data that will later be manipulated.


1b. Generating synthetic data
    **This is the next current task: On excel, create synthetic data, then save as CSV file and then implement.
Wave function specifics:

Wave 1:

A*sin(2pi*f*T)

Frequency: 60Hz (fundamental frequency)
period T = 1/10,000


Wave 2:
B * sin(2pi*180*T)

Combine:
Combined Signal = wave 1 + wave 2*0.3


2.) fft_analysis

-First run Fast Fourier transfer on created data
Log 1 for goal no.2: necessary data was able to be extracted from the original pool in the file (Time & Voltage), with the printing of extracted data from columns that
are of non-priority (wave1 & wave2)


-Next step is to extract data