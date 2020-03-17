#
#   ALL RIGHTS RESERVED.
#   _________________________________________________________________________________
#   NOTICE: All information contained  herein is, and remains the property  of  Varga
#   Consulting and  its suppliers, if  any. The intellectual and  technical  concepts
#   contained herein are proprietary to Varga Consulting and its suppliers and may be
#   covered  by  Canadian and  Foreign Patents, patents in process, and are protected
#   by  trade secret or copyright law. Dissemination of this information or reproduc-
#   tion  of  this  material is strictly forbidden unless prior written permission is
#   obtained from Varga Consulting.
#
#   Copyright © <2018> Varga Consulting, Toronto, On          info@vargaconsulting.ca
#   _________________________________________________________________________________

app = iex2h5

ARMA_CFLAGS = -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5 -DARMA_64BIT_WORD

LIBS =      -lglog  -lprofiler -lpcap -pthread -ltz -lcurl   		\
			-lboost_program_options -lprofiler						\
			-lboost_serialization -lboost_system 					\
			-lhdf5 -lm -lz  -ldl -lpthread 							\
			-llapack -lblas	-larmadillo		 						\

CXXFLAGS =  -I ../../include -I/usr/local/include  -std=c++17  		\
			-DARMA_DONT_USE_WRAPPER 								\
			-DHOST_NAME_MAX=254 									\
			-DVALUE_PRECISION=6 -DSTAT_PRECISION=4  				\
			-DDEBUG -DIEX_MAX_SYMBOLS=20000 -g -O0 -no-pie

%.o : $(SRC_DIR)/%.cpp 
	$(CXX) -$(INCLUDES) -o $@  $(CPPFLAGS) $(CXXFLAGS) -c $^

iex2h5: $(app).o consumer_asset.o consumer_irts.o consumer_rts.o consumer_index.o
	$(CXX) $^ $(LIBS) -o $@	

all: $(app) 

#PARSING directory
#------------------
tops_files :=$(sort $(wildcard /home/steven/Downloads/tops/*))
tops: $(app) iex.h5
	$(foreach dp, $(tops_files),  unpigz -c $(dp) | ./$(app) -g 9 --time-interval 60 --command irts; )

deep_files :=$(sort $(wildcard /home/steven/Downloads/deep/*))
deep: $(app) iex.h5
	$(foreach dp, $(deep_files),  unpigz -c $(dp) |  ./$(app) -g 9 --time-interval 60 --command rts; )


test-irts: $(app) iex.h5 
	unpigz -c ../../data/deeps.pcap.gz |  ./$(app) -g 9 --time-interval 60  --command irts

test-deeps: $(app) iex.h5
	unpigz -c ../../data/deeps.pcap.gz |  ./$(app) -g 9 --time-interval 60 --command rts
test-tops: $(app) iex.h5
	$(foreach echo dp, $(deep_files),  unpigz -c $(dp) |  
	unpigz -c ../../data/2018-02-01.pcap.gz  | ./$(app) -g 9   --time-interval 60  --command rts
iex.h5: $(app)
	unpigz -c ../../data/tops.pcap.gz | ./$(app) -g 9   --command assets

install:
	$(CP) $(app) /usr/local/bin/

clean:
	@$(RM) *.o *.prof  $(app)

prof-iex2h5:
	./iex2h5 -i ../../data/deeps.pcap.gz  -g 9 --time-interval 60  --command irts
.PHONY: test

