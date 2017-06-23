#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <bitset>
#include <ctime>

#include <omp.h>

#include "NTL/ZZX.h"

#include "FHE.h"
#include "EncryptedArray.h"

using namespace std;
using namespace NTL;

void fastPower(Ctxt &dataCtxt, long degree) {
       // Taken from eqtesting.cpp so that there are fewer includes
       if (degree == 1) return;
       Ctxt orig = dataCtxt;
       long k = NumBits(degree);
       long e = 1;
       for (long i = k - 2; i >= 0; i--) {
               Ctxt tmp1 = dataCtxt;
               tmp1.smartAutomorph(1L << e); // 1L << e computes 2^e
               dataCtxt.multiplyBy(tmp1);
               e = 2 * e;
               if (bit(degree, i)) {
                       dataCtxt.smartAutomorph(2);
                       dataCtxt.multiplyBy(orig);
                       e += 1;
               }
       }
}

void eqTest(Ctxt &resultCtxt, const Ctxt &queryCtxt, const Ctxt &dataCtxt,
           long degree) {
       Ctxt tempCtxt = dataCtxt;
       tempCtxt = dataCtxt;
       tempCtxt -= queryCtxt;
       fastPower(tempCtxt, degree);
       tempCtxt.negate();
       tempCtxt.addConstant(ZZ(1));
       resultCtxt = tempCtxt;
}

void treeMultHelper(Ctxt &resultCtxt, vector<Ctxt> &currentLayer, vector<Ctxt> &nextLayer) {
       unsigned long previousSize = currentLayer.size();
       if (previousSize == 0) {
               return;
       } else if (previousSize == 1) {
               resultCtxt = currentLayer[0];
               return;
       }
       nextLayer.resize((previousSize / 2 + previousSize % 2), resultCtxt);
   #pragma omp parallel for
       for (unsigned long i = 0; i < previousSize / 2; i++) {
               currentLayer[2 * i].multiplyBy(currentLayer[2 * i + 1]);
               nextLayer[i] = currentLayer[2 * i];
       }

       if (previousSize % 2 == 1) {
               nextLayer[nextLayer.size()-1] = (currentLayer[previousSize - 1]);
       }
       currentLayer.clear();
       treeMultHelper(resultCtxt, nextLayer, currentLayer);
}

void treeMult(Ctxt &resultCtxt, const vector<Ctxt> &ctxtVec) {
       if (ctxtVec.size() > 1) {
               vector <Ctxt> currentLayer, nextLayer;
               currentLayer = ctxtVec;
               nextLayer.clear();
               treeMultHelper(resultCtxt, currentLayer, nextLayer);
       } else {
//        cout << "Only 1 Ciphertext; No Multiplication Done." << endl;
               resultCtxt = ctxtVec[0];
       }
}

void oneTotalProduct(Ctxt &resultCtxt, const Ctxt &dataCtxt, const long wordLength, const EncryptedArray &ea) {

        long numWords = floor(ea.size() / wordLength);
        resultCtxt = dataCtxt;
        if (wordLength == 1) {
                return;
        }
        long shiftAmt = 1;

        // auto startTime = chrono::high_resolution_clock::now();
        // auto endTime = chrono::high_resolution_clock::now();
        // chrono::duration<double> timeTaken = endTime - startTime;

        while (shiftAmt < wordLength) {
                // startTime = chrono::high_resolution_clock::now();
                Ctxt tempCtxt = resultCtxt;
                ea.shift(tempCtxt, -shiftAmt);
                resultCtxt.multiplyBy(tempCtxt); // ctxt = ctxt * (ctxt << "shiftAmt")

                shiftAmt = 2 * shiftAmt;
                // endTime = chrono::high_resolution_clock::now();
                // timeTaken = endTime-startTime;
                // cout << shiftAmt << ", Time Taken: " << timeTaken.count() << endl;
        }
}

void makeMask(ZZX &maskPoly, const long shiftAmt, const bool invertSelection, const long wordLength, const EncryptedArray &ea) {
        vector<ZZX> maskVec, oneMaskVec;
        if(invertSelection) {
                maskVec.assign(shiftAmt, ZZX(1));
                oneMaskVec.assign(wordLength-shiftAmt, ZZX(0));
        } else {
                maskVec.assign(shiftAmt, ZZX(0));
                oneMaskVec.assign(wordLength-shiftAmt, ZZX(1));
        }
        maskVec.insert(maskVec.end(), oneMaskVec.begin(), oneMaskVec.end());

        vector<ZZX> fullMaskVec = maskVec;
        for(unsigned long i = 2*wordLength; i < ea.size(); i += wordLength) {
                fullMaskVec.insert(fullMaskVec.end(), maskVec.begin(), maskVec.end());
        }

        fullMaskVec.resize(ea.size(), ZZX(0));
        ea.encode(maskPoly, fullMaskVec);
}

void simdShift(Ctxt &ciphertextResult, Ctxt &ciphertextData, const long shiftAmt, const long wordLength, const EncryptedArray &ea) {
        Ctxt tempCiphertext = ciphertextData;
        if(shiftAmt > 0) {
                ZZX maskPoly;
                makeMask(maskPoly, shiftAmt, 0, wordLength, ea);

                ea.shift(tempCiphertext, shiftAmt);
                tempCiphertext.multByConstant(maskPoly);
        } else if (shiftAmt < 0) {
                ZZX maskPoly;
                makeMask(maskPoly, -shiftAmt, 0, wordLength, ea);

                tempCiphertext.multByConstant(maskPoly);
                ea.shift(tempCiphertext, shiftAmt);
        }
        ciphertextResult = tempCiphertext;
}


int main(int argc, char* argv[]) {

        if(argc != 5) {
                cerr << "Wrong inputs!";
                cerr << endl << "Inputs: Level m attrLength conjSize" << endl;
        }
        long p = 2;
        long r = 1;
        // long m = 31775; 32767
        // long L = 29; 31

        long m = atoi(argv[1]);
        long L = atoi(argv[2]);
//    double heuristicSecurity = (3 * eulerTot(m) * 7.2) / ((L + 1) * 22 * 4) - 110;

        // Timers
        auto startTime = chrono::high_resolution_clock::now();
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<double> timeTaken = endTime - startTime;
        float totalTime = 0;

        // FHE instance initialization
        FHEcontext context(m, p, r);
        buildModChain(context, L);
        ZZX F = context.alMod.getFactorsOverZZ()[0];

        FHESecKey secretKey(context);
        const FHEPubKey &publicKey = secretKey;
        secretKey.GenSecKey(64);
        addFrbMatrices(secretKey);
        addSome1DMatrices(secretKey);
        EncryptedArray ea(context, F);

        long numSlots = ea.size();
        long plaintextDegree = ea.getDegree();

        long wordLength = atoi(argv[3]);
        long queryLength = 17;
        long numWords = floor(numSlots / wordLength);
        long numEmpty = numSlots % wordLength;

        long conjSize = atoi(argv[4]);

        // Output parameters to log
        IndexSet allPrimes(0,context.numPrimes()-1);
        clog << m << ", " << L << ", " << context.logOfProduct(allPrimes)/log(2.0) << ", " << p << ", " << plaintextDegree << ", " << numSlots << ", ";

        //  NOTE: Not true anymore, less efficient
        //  /*
        //    Words are spaced out such that the first characters of each word are packed into the slots before the second characters and so on...
        //    i.e. a_1,b_1,...,z_1,a_2,b_2,...,z_2,...,a_l,b_l,...,z_l
        //     a = a_1a_2...a_l
        //     b = b_1b_2...b_l
        //     ...
        //     z = z_1z_2...z_l
        //
        //     NOTE: If wordLength is a divisor of numSlots, then slot rotations can be used.
        //  */

        // Process the strings, one attribute and one queryLength
        string attrString = "spaceship";
        cout << endl << "Attribute String: " << attrString << endl;

         vector<ZZX> plaintextAttr((unsigned long)numSlots - numEmpty, ZZX(0));

        for(unsigned long i = 0; i < attrString.length(); i++) {
                plaintextAttr[i] = char2Poly(attrString[i]);
        }
        for(unsigned long i = attrString.length(); i < wordLength; i++) {
                plaintextAttr[i] = char2Poly('#');
        }
        for(unsigned long i = wordLength; i < numSlots-numEmpty; i++) {
                plaintextAttr[i] = plaintextAttr[i%wordLength];
        }
        plaintextAttr.resize(numSlots, ZZX(0));

        // for (unsigned long i = 0; i < attrString.length(); i++) {
        //         plaintextAttr[i*numWords] = char2Poly(attrString[i]);
        // }
        // // "#" is the empty character symbol
        // for (unsigned long i = attrString.length(); i < wordLength; i++) {
        //         plaintextAttr[i*numWords] = char2Poly('#');
        // }
        // for (unsigned long j = 0; j < wordLength; j++) {
        //         for(unsigned long i = 1; i < numWords; i++) {
        //                 plaintextAttr[j*numWords+i] = plaintextAttr[j*numWords];
        //         }
        // }
        // plaintextAttr.resize(numSlots, ZZX(0));

        // "$" is the wildcard character symbol
        string queryString = "sp$$e";
        queryLength = queryString.length();
        cout << "Query Pattern: " << queryString << endl;

        vector<ZZX> plaintextQuery((unsigned long)numSlots - numEmpty, ZZX(0));
        vector<ZZX> plaintextE((unsigned long)numSlots - numEmpty, ZZX(0));
        vector<ZZX> plaintextConjunction((unsigned long) numSlots, long2Poly(rand()%(1L << 7)));

        for(unsigned long i = 0; i < queryString.length(); i++) {
                plaintextQuery[i] = char2Poly(queryString[i]);
                if(queryString[i] == '$') {
                        plaintextE[i] = 1;
                }
        }
        for(unsigned long i = queryString.length(); i < wordLength; i++) {
                plaintextQuery[i] = char2Poly('$');
                plaintextE[i] = 1;
        }
        for(unsigned long i = wordLength; i < numSlots-numEmpty; i++) {
                plaintextQuery[i] = plaintextQuery[i%wordLength];
                plaintextE[i] = plaintextE[i%wordLength];
        }
        plaintextQuery.resize(numSlots, ZZX(0));
        plaintextE.resize(numSlots, ZZX(0));

        // for (unsigned long i = 0; i < queryString.length(); i++) {
        //         plaintextQuery[i*numWords] = char2Poly(queryString[i]);
        //         if (queryString[i] == '$') {
        //                 plaintextE[i*numWords] = 1;
        //         }
        // }
        // for (unsigned long j = 0; j < wordLength; j++) {
        //         for(unsigned long i = 1; i < numWords; i++) {
        //                 plaintextQuery[j*numWords+i] = plaintextQuery[j*numWords];
        //                 plaintextE[j*numWords+i] = plaintextE[j*numWords];
        //         }
        // }
        // plaintextQuery.resize(numSlots, ZZX(0));
        // plaintextE.resize(numSlots, ZZX(0));

        vector<ZZX> plaintextResult((unsigned long)numSlots - numEmpty, ZZX(1));
        plaintextResult.resize(numSlots, ZZX(0));

        // for(unsigned long i = wordLength; i < 2*wordLength; i++) {
        //     cout << poly2Char(plaintextAttr[i]) << ", ";
        // }
        // cout << endl;
        // for(unsigned long i = wordLength; i < 2*wordLength; i++) {
        //     cout << poly2Char(plaintextQuery[i]) << ", ";
        // }
        // cout << endl;
        // for(unsigned long i = wordLength; i < 2*wordLength; i++) {
        //     cout << plaintextE[i] << ", ";
        // }
        // cout << endl;

        clog << wordLength << ", " << queryLength << ", " << numWords << ", ";

        cout << "Plaintext Processing Done!" << endl;

        // Initialize and encrypt ciphertexts
        Ctxt ciphertextAttr(publicKey);
        Ctxt ciphertextQuery(publicKey);
        Ctxt tempCiphertext(publicKey);
        Ctxt ciphertextE(publicKey);
        Ctxt ciphertextResult(publicKey);
        Ctxt conjResult(publicKey);
        Ctxt conjQuery(publicKey);
        Ctxt conjCtxt(publicKey);

        ea.encrypt(ciphertextAttr, publicKey, plaintextAttr);
        ea.encrypt(ciphertextQuery, publicKey, plaintextQuery);
        ea.encrypt(ciphertextE, publicKey, plaintextE);
        ea.encrypt(conjCtxt, publicKey, plaintextConjunction);
        ea.encrypt(conjQuery, publicKey, plaintextConjunction);

        Ctxt oneCiphertext(publicKey);
        Ctxt zeroCiphertext(publicKey);

        vector<ZZX> onePlaintext(numSlots, ZZX(1));

        ea.encrypt(oneCiphertext, publicKey, onePlaintext);
        zeroCiphertext = oneCiphertext;
        zeroCiphertext -= oneCiphertext;

        // Initialize components
        vector<Ctxt> ciphertextWs, ciphertextRs, ciphertextSs, ciphertextDs;
        for (unsigned long i = 0; i < wordLength; i++) {
                ciphertextWs.push_back(ciphertextAttr);
                ciphertextRs.push_back(ciphertextAttr);
                ciphertextSs.push_back(ciphertextAttr);
        }

        for (unsigned long i = 0; i < wordLength; i++) {
                if (i <= wordLength - queryLength) {
                        ciphertextDs.push_back(oneCiphertext);
                } else {
                        ciphertextDs.push_back(zeroCiphertext);
                }
        }

        // For compound conjunction queries
        vector<Ctxt> ciphertextConj;
        vector<Ctxt> ciphertextConjResult;
        for(unsigned long i = 0; i < conjSize; i++) {
                ciphertextConj.push_back(ciphertextAttr);
                ciphertextConjResult.push_back(ciphertextAttr);
        }

        cout << "Encryption Done!" << endl;

        ZZX selectPoly;
        ZZX queryMask, finalMask;
        makeMask(selectPoly, 1, 1, wordLength, ea);
        makeMask(finalMask, wordLength, 1, wordLength, ea);

        // Step 1: Shift the attributes
    //     Remnants of experiment to pack differently, all characters of the same slot first
    //     But the shift time is much longer than packing word by word

    //     startTime = chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    //     for(unsigned long i = 1; i < ciphertextWs.size(); i++) {
    //             ea.shift(ciphertextWs[i],-i*numWords);
    //     }
    //     endTime = chrono::high_resolution_clock::now();
    //     timeTaken = endTime-startTime;
    //     totalTime += timeTaken.count();
    //     cout << "Pre-compute Time: " << timeTaken.count() << endl;

        startTime = chrono::high_resolution_clock::now();
    #pragma omp parallel for
        for(unsigned long i = 1; i < ciphertextWs.size(); i++) {
                simdShift(ciphertextWs[i], ciphertextAttr, -i, wordLength, ea);
                // ea.shift(ciphertextWs[i],-i);
        }
        endTime = chrono::high_resolution_clock::now();
        timeTaken = endTime-startTime;
        // totalTime += timeTaken.count();
        cout << "Pre-compute Time: " << timeTaken.count() << endl;
        clog << timeTaken.count() << ", ";

        // for(unsigned long i = 0; i < ciphertextWs.size(); i++) {
        //         ea.decrypt(ciphertextWs[i],secretKey,plaintextResult);
        //         cout << poly2Char(plaintextResult[0]) << ", ";
        // }
        // cout << endl;

        // Step 2: Test if the characters are the same
        startTime = chrono::high_resolution_clock::now();
    #pragma omp parallel for
        for(unsigned long i = 0; i < ciphertextWs.size()+conjSize; i++) {
                if(i < ciphertextWs.size()) {
                        ciphertextWs[i] += ciphertextQuery;
                } else {
                        ciphertextConj[i-ciphertextWs.size()] += ciphertextAttr;
                }
        }
        endTime = chrono::high_resolution_clock::now();
        timeTaken = endTime-startTime;
        totalTime += timeTaken.count();
        cout << "XOR Time: " << timeTaken.count() << endl;

        startTime = chrono::high_resolution_clock::now();
    #pragma omp parallel for
        for(unsigned long i = 0; i < ciphertextWs.size()+conjSize; i++) {
                if(i < ciphertextWs.size()) {
                        equalTest(ciphertextRs[i], zeroCiphertext, ciphertextWs[i], plaintextDegree);
                } else {
                        equalTest(ciphertextConjResult[i-ciphertextWs.size()], zeroCiphertext, ciphertextConj[i-ciphertextWs.size()], plaintextDegree);
                }
        // }
        // for(unsigned long i = 0; i < ciphertextRs.size(); i++) {
            if(i < ciphertextWs.size()) {
                ciphertextRs[i] += ciphertextE;
            }

        }
        endTime = chrono::high_resolution_clock::now();
        timeTaken = endTime-startTime;
        totalTime += timeTaken.count();
        cout << "Level left: " << ciphertextRs[1].findBaseLevel() << ", Equality Check + eMask Time: " << timeTaken.count() << endl;
        clog << timeTaken.count() << ", ";

        // Step 3: Combine results of character tests per shift
        startTime = chrono::high_resolution_clock::now();
    #pragma omp parallel for
        for(unsigned long i = 0; i < ciphertextRs.size()+conjSize; i++) {
                if(i < ciphertextRs.size()) {
                        oneTotalProduct(ciphertextSs[i], ciphertextRs[i], wordLength, ea);
                } else if (conjSize > 0) {
                        oneTotalProduct(ciphertextConj[i-ciphertextRs.size()], ciphertextConjResult[i-ciphertextRs.size()], wordLength, ea);
                }
        }
        endTime = chrono::high_resolution_clock::now();
        timeTaken = endTime-startTime;
        totalTime += timeTaken.count();
        cout << "Level left: " << ciphertextSs[1].findBaseLevel() << ", Product of Equalities: " << timeTaken.count() << endl;
        clog << timeTaken.count() << ", ";

        // Step 4: Combine results from testing every possible shift
        startTime = chrono::high_resolution_clock::now();
    #pragma omp parallel for
        for(unsigned long i = 0; i < ciphertextSs.size(); i++) {
                ciphertextSs[i].multiplyBy(ciphertextDs[i]);
                ciphertextSs[i].addConstant(selectPoly);
        }
        endTime = chrono::high_resolution_clock::now();
        timeTaken = endTime-startTime;
        totalTime += timeTaken.count();
        cout << "Level left: " << ciphertextSs[1].findBaseLevel() << ", Disjunction Prep Time: " << timeTaken.count() << endl;
        clog << timeTaken.count() << ", ";

        startTime = chrono::high_resolution_clock::now();
        treeMult(ciphertextResult, ciphertextSs);
        ciphertextResult.addConstant(selectPoly);
        treeMult(conjResult, ciphertextConj);
        if(conjSize > 0) {
                ciphertextResult.multiplyBy(conjResult);
        }
        // ciphertextResult.multByConstant(finalMask);
        endTime = chrono::high_resolution_clock::now();
        timeTaken = endTime-startTime;
        totalTime += timeTaken.count();
        cout << "Level left: " << ciphertextResult.findBaseLevel() << ", Disjunction Time: " << timeTaken.count() << endl;
        clog << timeTaken.count() << ", ";

        clog << totalTime << ", ";

        startTime = chrono::high_resolution_clock::now();
        tempCiphertext = ciphertextResult;
    #pragma omp parallel for
        for(unsigned long i = 1; i < wordLength; i++) {
                ciphertextWs[i] = ciphertextResult;
                ea.shift(ciphertextWs[i], i);
        }
        for(unsigned long i = 1; i < wordLength; i++) {
                tempCiphertext += ciphertextWs[i];
        }
        tempCiphertext.multiplyBy(ciphertextAttr);
        endTime = chrono::high_resolution_clock::now();
        timeTaken = endTime-startTime;
        // totalTime += timeTaken.count();
        cout << "Level left: " << tempCiphertext.findBaseLevel() << ", Fill + Selection Time: " << timeTaken.count() << endl;
        clog << timeTaken.count() << ", ";

        cout << "Total Time: " << totalTime << endl;

        ea.decrypt(tempCiphertext, secretKey, plaintextResult);
        for(unsigned long i = 0; i < wordLength; i++) {
                cout << poly2Char(plaintextResult[i]) << ", ";
        }
        cout << endl;

        ea.decrypt(conjResult, secretKey, plaintextResult);
        for(unsigned long i = 0; i < wordLength; i++) {
            cout << poly2Long(plaintextResult[i]) << ", ";
        }
        cout << endl;

        for(unsigned long i = 0; i < conjSize; i++) {
            ea.decrypt(ciphertextConj[i],secretKey, plaintextResult);
            for(unsigned long i = 0; i < wordLength; i++) {
                cout << poly2Long(plaintextResult[i]) << ", ";
            }
            cout << endl;
        }
        cout << endl;
        clog << endl;
}
