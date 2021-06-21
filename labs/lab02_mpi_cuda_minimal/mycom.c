#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include "mycom.h"

int iabs(int a)
{
  if (a>=0)
    return a;
  else
    return (-a);
}

int imax(int a, int b)
{
  if (a>=b)
    return a;
  else
    return b;
}

int imin(int a, int b)
{
  if (a<=b)
    return a;
  else
    return b;
}

double dabs(double a)
{
  if (a>=0)
    return a;
  else
    return (-a);
}

double dmax(double a, double b)
{
  if (a>=b)
    return a;
  else
    return b;
}

double dmin(double a, double b)
{
  if (a<=b)
    return a;
  else
    return b;
}

double dsin(double x) {
  double s = sin(x);
  if (dabs(s)>1e-15)
    return s;
  else
    return 0.0;
}

double dcos(double x) {
  double s = cos(x);
  if (dabs(s)>1e-15)
    return s;
  else
    return 0.0;
}

double dexp(double x) {
  double s = exp(x);
  if (dabs(s)>1e-15)
    return s;
  else
    return 0.0;
}

void MyErr(const char *msg, const int n)
{
  puts(msg); exit(n);
}

double MyDelay(const int n)
{
  int i;
  double s = 0;

  for (i=0; i<n; i++) {
    s += exp(-1.23*i);
  }

  return (s/n);
}

int fopen_m(FILE** F, const char* name, const char* mode)
{
  int i;

  *F = fopen(name, mode);

  if (*F == NULL) i = ENOENT; else i = ferror(*F);

  if (i != 0)
   switch (i)
    {
     case ENOENT:
      {
       fprintf(stderr,"Input file not found !\n");
       return EC_Bad_file_name;
      }
     case EMFILE:
      {
       fprintf(stderr,"Too many opened files !\n");
       return EC_Many_open_files;
      }
     case EACCES:
      {
       fprintf(stderr,"Access to input file is not correct !\n");
       return EC_Bad_file_access;
      }
     default:
      {
       fprintf(stderr,"Fatal i/o error !\n");
       return EC_Fatal_error;
      }
    }

  return 0;
}

int fclose_m(FILE **F)
{
  if (fclose(*F) != 0)
  {
   fprintf(stderr,"File close error !\n");
   return EC_Close_file_error;
  }
  return 0;
}

int FReadString(FILE* F, char* str, const int maxlen)
{
 int k, l;
 char *p;

 do {
   p = fgets(str, maxlen, F);
   k = ferror(F);
   if (k != 0) return EC_File_reading_error;

   l = strlen(str);
   if (l > 0 && str[0] == '#') l = 0; // comments in file
 } while ( p && (l == 0) && feof(F) == 0 );

 while (l > 0 && str[l-1] == '\n') l--;

 if (l > 0) str[l] = 0;

 return 0;
}

int fget_int(FILE* F, const int l, int* a)
{
  int i;
  char sbuf[256];

  if (l<1 || l>255) return -1;
  if (F == NULL) return -2;
  if (feof(F)) return -3;

  i = FReadString(F, sbuf, 256);
  if (i != 0) return i;

  sbuf[l] = 0;
  i = sscanf(sbuf,"%d\n",a);
  if (i != 1) return -4;

  return 0;
}

int fget_dbl(FILE* F, const int l, double* a)
{
  int i;
  char sbuf[256];

  if (l<1 || l>255) return -1;
  if (F == NULL) return -2;
  if (feof(F)) return -3;

  i = FReadString(F, sbuf, 256);
  if (i != 0) return i;

  sbuf[l] = 0;
  i = sscanf(sbuf,"%le\n",a);
  if (i != 1) return -4;

  return 0;
}
