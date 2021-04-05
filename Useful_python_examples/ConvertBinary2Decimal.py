# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:31:38 2020

@author: Mingming
"""


def Bin2Dec(byte0, byte1, byte2):
    '''
    
    Convert binary data into Decimal data
    
    In some cases, when we return binary data into serial port,
    it reads like "b'\xb0\x9f\x00\x00\xe0\xa1\x00\x00\x80\x9d\x00\x00\xf0\xa0\x00\x00 \x9e\x00..."
    The first four bytes are "b0", "9f","00","00", they are arranged in the order from LSB to MSB.
    
    Below is the function to conver the binary data into signed decimal numbers. 
    

    Parameters
    ----------
    byte0 : binary data, 
    byte1 : binary data, 
    byte2 : binary data, 
    byte3 : binary data, 
        From byte0 to byte3 is from the Least Significant Byte to Most Significant Byte.

    Returns
    -------
    my_signed_number : signed decimal number
        .

    '''
    
    
    total_binary = byte2<<16|byte1<<8|byte0   # LSB is on the further right

    binary_bits = bin(total_binary)

    if binary_bits[0:2] == '0b':
        real_bytes = binary_bits[2:]
    else:
        real_bytes = binary_bits
    
    # the most significant byte might be less than 4 bits, then the string will only have
    # one or two bits for displaying (up to the bit with value of '1')
    # Thus, the first bit here is not the sign bit. 
    if (len(real_bytes)%4) == 0 and len(real_bytes)==32:
        sign_bit = real_bytes[0] # the first bit is the first bit of actural number                                
    else:
        sign_bit = '0'
    
    # this is a positive number
    if sign_bit == '0':
       my_signed_number = int(real_bytes, 2) 
    # this is a negative number
    elif sign_bit == '1':
         # conduct two's complement
         output_bits = reverse_bits(real_bytes)  
         my_signed_number = - ( int(output_bits, 2) + 1 )
  
    
    return my_signed_number
    
  

def reverse_bits(input_bits): 
    """
    For a binary input data, reverse each bit. 
    Usage:
        output_bits = reverse_bits(input_bits)
    
    INPUT:
        input_bits: the input binary data, a string
    OUTPUT:
        output_bits: the output binary data with each bit reversed,
        a string.        
    """   
    output_bits = ""
    for bit in input_bits:        
        output_bits += reverse_1_bit(bit)
    return output_bits  


def reverse_1_bit(input_bit): 
    """
    Given one bit of binary data, reverse this bit. 
    """     
    if input_bit == '1':
        output_bit = '0'
    elif input_bit == '0':
        output_bit = '1'
    return output_bit