C hyperelastic.f
C
C User subroutine VUMAT
      subroutine vumat (
C Read only -
     *     nblock, ndir, nshr, nstatev, nfieldv, nprops, lanneal,
     *     stepTime, totalTime, dt, cmname, coordMp, charLength,
     *     props, density, strainInc, relSpinInc,
     *     tempOld, stretchOld, defgradOld, fieldOld,
     *     stressOld, stateOld, enerInternOld, enerInelasOld,
     *     tempNew, stretchNew, defgradNew, fieldNew,
C Write only -
     *     stressNew, stateNew, enerInternNew, enerInelasNew )
C
      include 'vaba_param.inc'
C
      dimension coordMp(nblock,*), charLength(nblock), props(nprops),
     1     density(nblock), strainInc(nblock,ndir+nshr),
     2     relSpinInc(nblock,nshr), tempOld(nblock),
     3     stretchOld(nblock,ndir+nshr), 
     4     defgradOld(nblock,ndir+nshr+nshr),
     5     fieldOld(nblock,nfieldv), stressOld(nblock,ndir+nshr),
     6     stateOld(nblock,nstatev), enerInternOld(nblock),
     7     enerInelasOld(nblock), tempNew(nblock),
     8     stretchNew(nblock,ndir+nshr),
     9     defgradNew(nblock,ndir+nshr+nshr),
     1     fieldNew(nblock,nfieldv),
     2     stressNew(nblock,ndir+nshr), stateNew(nblock,nstatev),
     3     enerInternNew(nblock), enerInelasNew(nblock)
C     
      character*80 cmname
      dimension intv(2), B(5), F(5), lamdastar(3)
      real J, p, nu, M, N, A, I1,coe,coe2
      parameter ( zero = 0.d0, one = 1.d0, two = 2.d0, three = 3.d0,
     *     third = one / three, half = 0.5d0, twothds = two / three,
     *     op5 = 1.5d0, fivesix = 5.6d0, twoes = 2.87d0, delta=1.001 )
      parameter ( tempFinal = 1.d2, timeFinal = 1.d-2 )
C
C     
*
*     Check that ndir=3 and nshr=1. If not, exit.
*
      intv(1) = ndir
      intv(2) = nshr
      if (ndir .ne. 3 .or. nshr .ne. 1) then
         call xplb_abqerr(1,'Subroutine VUMAT is implemented '//
     *        'only for plane strain and axisymmetric cases '//
     *        '(ndir=3 and nshr=1)',0,zero,' ')
         call xplb_abqerr(-2,'Subroutine VUMAT has been called '//
     *        'with ndir=%I and nshr=%I',intv,zero,' ')
         call xplb_exit
      end if
*     
      nu =  props(1)
      A  = props(2)
      N  = props(3)
      M  =props(4)

      
*
        
      do k = 1, nblock
      
                F(1)=defgradNew(k,1)
                F(2)=defgradNew(k,2)
                F(3)=defgradNew(k,3)
                F(4)=defgradNew(k,4)
                F(5)=defgradNew(k,5)
				
C	   		WRITE (*,*) 'F4', F(4)                

               J=F(1)*F(2)*F(3)-F(4)*F(5)*F(3)
C              t1=J**(-twothds)
			  
C	Calculate B
              
              B(1)=(F(1)**two+F(5)**two)
              B(2)=(F(4)**two+F(2)**two)
              B(3)=(F(3)**two)
              B(4)=(F(1)*F(4)+F(2)*F(5))
              B(5)=B(4)
              
              I1=(B(1)+B(2)+B(3))/3
              coe=nu/J
			  coe2=J**(-twothds)
              

	             p=A*(J**(-N-1)-J**(-M-1))
         
               
              stressNew(k,1) = coe*coe2*(B(1)-I1)-p
              stressNew(k,2) = coe*coe2*(B(2)-I1)-p
              stressNew(k,3) = coe*coe2*(B(3)-I1)-p
              stressNew(k,4) = coe*coe2*B(4)
          
        end do
      
*     
      return
      end
	  
	  
	  
	  
	  
	  
        subroutine vuinter(
c Write only
     1 sfd, scd, spd, svd,
C Read/Write - 
     2 stress, fluxSlv, fluxMst, sed, statev,
C Read only - 
     3 kStep, kInc, nFacNod, nSlvNod, nMstNod, nSurfDir,
     4 nDir, nStateVar, nProps, nTemp, nPred, numDefTfv,
     5 jSlvUid, jMstUid, jConMstid, timStep, timGlb,
     6 dTimCur, surfInt, surfSlv, surfMst,
     7 rdisp, drdisp, drot, stiffDflt, condDflt,
     8 shape, coordSlv, coordMst, alocaldir, props,
     9 areaSlv, tempSlv, dtempSlv, preDefSlv, dpreDefSlv,
     1 tempMst, dtempMst, preDefMst, dpreDefMst) 
C
      include 'vaba_param.inc'
C
      character*80 surfInt, surfSlv, surfMst
	  
	 
C
      dimension props(nProps), statev(nStateVar,nSlvNod), 
     1 drot(2,2,nSlvNod), sed(nSlvNod), sfd(nSlvNod),
     2 scd(nSlvNod), spd(nSlvNod), svd(nSlvNod),
     3 rdisp(nDir,nSlvNod), drdisp(nDir,nSlvNod),
     4 stress(nDir,nSlvNod), fluxSlv(nSlvNod),
     5 fluxMst(nSlvNod), areaSlv(nSlvNod),
     6 stiffDflt(nSlvNod), condDflt(nSlvNod),
     7 alocaldir(nDir,nDir,nSlvNod), shape(nFacNod,nSlvNod),
     8 coordSlv(nDir,nSlvNod), coordMst(nDir,nMstNod),
     9 jSlvUid(nSlvNod), jMstUid(nMstNod),
     1 jConMstid(nFacNod,nSlvNod), tempSlv(nSlvNod),
     2 dtempSlv(nSlvNod), preDefSlv(nPred,nSlvNod),
     3 dpreDefSlv(nPred,nSlvNod), tempMst(numDefTfv),
     4 dtempMst(numDefTfv), preDefMst(nPred,numDefTfv),
     5 dpreDefMst(nPred,numDefTfv)
	 
	 REAL(8) ::  zero, one, two, third, half, toler, pi
	 REAL(8) :: stressc,deltac,kcontact,rel
	 REAL(8) :: kSlv, deltaf,fiveth, four, ten
	 
	  parameter( cutoff = 2.d0, zero = 0.d0, one = 1.d0, two = 2.d0, three = 3.,
     1  four = 4.d0,five = 5.d0, pi=3.1416d0, half = .5d0, ten=10.d0)

C user coding to define stress,
C      and, optionally, fluxSlv, fluxMst, statev, sed, sfd, scd, spd,
C      and svd
c
c Local variables
       stressc = 220E6
       deltac = 0.01E-3
	   toler = 1.0E-8
	   kcontact= 1.0E15
	   rel = 1.0E-15
       deltaf = 0.05E-3
C	   		WRITE (*,*) 'gap', rdisp(1,kSlv)


C	define normal cohesive behavior
      do kSlv = 1, nSlvNod  
	    h=-rdisp(1,kSlv)
C h<0 means peneration; h>0 means gap
	       if ( (h.le. deltac) .AND. (h.ge. toler)) then     
               stress(1,kSlv) = -stressc/deltac*h       
           else if ( (h.ge. deltac) .AND. (h.le. deltaf)) then
			   stress(1,kSlv) = stressc/(deltaf-deltac)*(h-deltaf)
C           else if (h.le. rel) then
C			   stress(1,kSlv) = -(kcontact)*(h-rel)		 
		   else
               stress(1,kSlv)=zero
	       end if
C		  WRITE (*,*) 'stress', stress(1,kSlv)

C	define tangential cohesive behavior
C	    s=-rdisp(2,kSlv)
C	       if ( (s.le. deltac) .AND. (s.ge. toler)) then     
C               stress(2,kSlv) = -stressc/deltac*s       
C           else if ( (s.ge. deltac) .AND. (s.le. deltaf)) then
C			   stress(2,kSlv) = stressc/(deltaf-deltac)*(s-deltaf) 
C		   else
C               stress(2,kSlv)=zero
C	       end if
C		  WRITE (*,*) 'stress', stress(2,kSlv)


      end do
	  
      return
      end
	  