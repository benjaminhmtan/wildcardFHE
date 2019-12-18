#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>
#include <bitset>
#include <ctime>

#include <omp.h>

#include "NTL/ZZX.h"

#include "helib/FHE.h"
#include "helib/EncryptedArray.h"

#define new_simd 0

NTL::ZZX long2Poly(long query) {
    NTL::ZZX queryPoly;
    queryPoly.SetLength(64);
    std::bitset<64> queryBits = query;

    for (int i = 0; i < queryBits.size(); i++) {
        NTL::SetCoeff(queryPoly, i, queryBits[i]);
    }
    queryPoly.normalize();

    return queryPoly;
}

NTL::ZZX char2Poly(char character) {
    int charCode = character;
    NTL::ZZX resultPoly;

    resultPoly = long2Poly((long) charCode);

    return resultPoly;
}

long poly2Long(NTL::ZZX result) {
    long resultLong = 0;
    for (int i = 0; i <= deg(result); i++) {
        resultLong += (1L << i) * (NTL::coeff(result, i) == NTL::ZZ(1));
    }
    return resultLong;
}

char poly2Char(NTL::ZZX result) {
    char character;

    character = (int) poly2Long(result);

    return character;
}

void fastPower(Ctxt &dataCtxt, long degree) {
    // Taken from eqtesting.cpp so that there are fewer includes
    if (degree == 1) return;
    Ctxt orig = dataCtxt;
    long k = NTL::NumBits(degree);
    long e = 1;
    for (long i = k - 2; i >= 0; i--) {
        Ctxt tmp1 = dataCtxt;
        tmp1.smartAutomorph(1L << e); // 1L << e computes 2^e
        dataCtxt.multiplyBy(tmp1);
        e = 2 * e;
        if (NTL::bit(degree, i)) {
            dataCtxt.smartAutomorph(2);
            dataCtxt.multiplyBy(orig);
            e += 1;
        }
    }
}

void equalTest(Ctxt &resultCtxt, const Ctxt &queryCtxt, const Ctxt &dataCtxt,
               long degree) {
    Ctxt tempCtxt = dataCtxt;
    tempCtxt = dataCtxt;
    tempCtxt -= queryCtxt;
    fastPower(tempCtxt, degree);
    tempCtxt.negate();
    tempCtxt.addConstant(NTL::ZZ(1));
    resultCtxt = tempCtxt;
}

void treeMultHelper(Ctxt &resultCtxt, std::vector<Ctxt> &currentLayer, std::vector<Ctxt> &nextLayer) {
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
        nextLayer[nextLayer.size() - 1] = (currentLayer[previousSize - 1]);
    }
    currentLayer.clear();
    treeMultHelper(resultCtxt, nextLayer, currentLayer);
}

void treeMult(Ctxt &resultCtxt, const std::vector<Ctxt> &ctxtVec) {
    if (ctxtVec.size() > 1) {
        std::vector<Ctxt> currentLayer, nextLayer;
        currentLayer = ctxtVec;
        nextLayer.clear();
        treeMultHelper(resultCtxt, currentLayer, nextLayer);
    } else {
//        // std::cout << "Only 1 Ciphertext; No Multiplication Done." << std::endl;
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

    // auto startTime = std::chrono::high_resolution_clock::now();
    // auto endTime = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> timeTaken = endTime - startTime;

    while (shiftAmt < wordLength) {
        // startTime = std::chrono::high_resolution_clock::now();
        Ctxt tempCtxt = resultCtxt;
#if new_simd
        ea.shift(tempCtxt, (-shiftAmt*numWords));
#else
         ea.shift(tempCtxt, (-shiftAmt));
#endif
        resultCtxt.multiplyBy(tempCtxt); // ctxt = ctxt * (ctxt << "shiftAmt")

        shiftAmt = 2 * shiftAmt;
        // endTime = std::chrono::high_resolution_clock::now();
        // timeTaken = endTime-startTime;
        // // std::cout << shiftAmt << ", Time Taken: " << timeTaken.count() << std::endl;
    }
}

void makeMask(NTL::ZZX &maskPoly, const long shiftAmt, const bool invertSelection, const long wordLength,
              const EncryptedArray &ea) {
    std::vector<NTL::ZZX> maskVec, oneMaskVec;
    if (invertSelection) {
        maskVec.assign(shiftAmt, NTL::ZZX(1));
        oneMaskVec.assign(wordLength - shiftAmt, NTL::ZZX(0));
    } else {
        maskVec.assign(shiftAmt, NTL::ZZX(0));
        oneMaskVec.assign(wordLength - shiftAmt, NTL::ZZX(1));
    }
    maskVec.insert(maskVec.end(), oneMaskVec.begin(), oneMaskVec.end());

    std::vector<NTL::ZZX> fullMaskVec = maskVec;
    for (unsigned long i = 2 * wordLength; i < ea.size(); i += wordLength) {
        fullMaskVec.insert(fullMaskVec.end(), maskVec.begin(), maskVec.end());
    }

    fullMaskVec.resize(ea.size(), NTL::ZZX(0));
    ea.encode(maskPoly, fullMaskVec);
}

void simdShift(Ctxt &ciphertextResult, Ctxt &ciphertextData, const long shiftAmt, const long wordLength,
               const EncryptedArray &ea) {
    Ctxt tempCiphertext = ciphertextData;
    if (shiftAmt > 0) {
        NTL::ZZX maskPoly;
        makeMask(maskPoly, shiftAmt, 0, wordLength, ea);

        ea.shift(tempCiphertext, shiftAmt);
        tempCiphertext.multByConstant(maskPoly);
    } else if (shiftAmt < 0) {
        NTL::ZZX maskPoly;
        makeMask(maskPoly, -shiftAmt, 0, wordLength, ea);

        tempCiphertext.multByConstant(maskPoly);
        ea.shift(tempCiphertext, shiftAmt);
    }
    ciphertextResult = tempCiphertext;
}


int main(int argc, char *argv[]) {

    if (argc != 5) {
        std::cerr << "Wrong inputs!";
        std::cerr << std::endl << "Inputs: level m attrLength conjSize" << std::endl;
        return 1;
    }
    long p = 2;
    long r = 1;
    // long m = 31775; 32767
    // long L = 29; 31

    long m = atoi(argv[2]);
    long L = atoi(argv[1]);
//    double heuristicSecurity = (3 * eulerTot(m) * 7.2) / ((L + 1) * 22 * 4) - 110;

    // Define wildcard charaters
    char blankChar = 2;
    char wcChar = 3;
    char excludeChar = 4;

    // Timers
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timeTaken = endTime - startTime;
    float totalTime = 0;

    // FHE instance initialization
    FHEcontext context(m, p, r);
    buildModChain(context, L);
    NTL::ZZX F = context.alMod.getFactorsOverZZ()[0];

    FHESecKey secretKey(context);
    const FHEPubKey &publicKey = secretKey;
    secretKey.GenSecKey(64);
    addFrbMatrices(secretKey);
    addBSGS1DMatrices(secretKey);
    EncryptedArray ea(context, F);

    long numSlots = ea.size();
    long plaintextDegree = ea.getDegree();

    long wordLength = atoi(argv[3]);
    long queryLength = 17;
    long numWords = floor(numSlots / wordLength);
    long numEmpty = numSlots % wordLength;

    long conjSize = atoi(argv[4]);

    // Output parameters to log
    IndexSet allPrimes(0, context.numPrimes() - 1);
    std::clog << m << ", " << L << ", " << context.logOfProduct(context.ctxtPrimes) / log(2.0) << ", " <<  context.logOfProduct(allPrimes) / log(2.0) << ", " << p << ", " << plaintextDegree
         << ", " << numSlots << ", ";

    // Process the strings, one attribute and one query pattern of type %W%,
    // * wildcard is encoded as ASCII code 2
    // # wildcard is encoded as ASCII code 3

    std::string attrString = "spares";
    // std::cout << std::endl << "Attribute String: " << attrString << std::endl;

    std::vector<NTL::ZZX> plaintextAttr;
#if new_simd
    for(long i = 0; i < attrString.length(); i++)
        plaintextAttr.resize((i+1)*numWords, char2Poly(attrString[i]));
    for(long i = attrString.length(); i < wordLength; i++)
        plaintextAttr.resize((i+1)*numWords, plaintextAttr[i] = char2Poly(blankChar));
#else
    plaintextAttr.resize(numSlots-numEmpty, NTL::ZZX(0));
    for (unsigned long i = 0; i < attrString.length(); i++) 
        plaintextAttr[i] = char2Poly(attrString[i]);
    for (unsigned long i = attrString.length(); i < wordLength; i++)
        plaintextAttr[i] = char2Poly(blankChar);
    for (unsigned long i = wordLength; i < numSlots - numEmpty; i++)
        plaintextAttr[i] = plaintextAttr[i % wordLength];    
#endif
    plaintextAttr.resize(numSlots, NTL::ZZX(0));

    // "$" is the wildcard character symbol
    std::string queryString = "sp";
    queryString += wcChar;
    queryString += excludeChar;
    queryString += "c";
    queryString += "e";
    queryLength = queryString.length();
    // std::cout << "Query Pattern: " << queryString << std::endl;

    std::vector<NTL::ZZX> plaintextQuery, plaintextE;
    std::vector<NTL::ZZX> plaintextConjunction((unsigned long) numSlots, long2Poly(rand() % (1L << 7)));

#if new_simd
    int counter = 0;
    for (unsigned long i = 0; i < queryString.length(); i++) {
        if (queryString[i] == wcChar) {
            plaintextQuery.resize((i+1)*numWords, char2Poly(queryString[i]));
            plaintextE.resize((counter+1)*numWords, NTL::ZZX(1));
            counter++;
        } else if (queryString[i] == excludeChar) {
            plaintextQuery.resize((i+1)*numWords, char2Poly(queryString[i+1]));
            plaintextE.resize((counter+1)*numWords, NTL::ZZX(1));
            // plaintextQuery[counter] =
            i += 1;
            counter++;
        } else {
            plaintextQuery.resize((i+1)*numWords, char2Poly(queryString[i]));
            plaintextE.resize((counter+1)*numWords, NTL::ZZX(0));
            plaintextQuery[counter] = char2Poly(queryString[i]);
            counter++;
        }
        // // std::cout << i << ", " << counter << std::endl;
    }
    for (unsigned long i = counter; i < wordLength; i++) {
        plaintextQuery.resize((i+1)*numWords, char2Poly(wcChar));
        plaintextE.resize((i+1)*numWords, NTL::ZZX(1));
    }
    
#else
    plaintextQuery.resize(numSlots-numEmpty, NTL::ZZX(0));
    plaintextE.resize(numSlots-numEmpty, NTL::ZZX(0));
    int counter = 0;
    for (unsigned long i = 0; i < queryString.length(); i++) {
        if (queryString[i] == wcChar) {
            plaintextQuery[counter] = char2Poly(excludeChar);
            plaintextE[counter] = 1;
            counter++;
        } else if (queryString[i] == excludeChar) {
            plaintextQuery[counter] = char2Poly(queryString[i + 1]);
            plaintextE[counter] = 1;
            i += 1;
            counter++;
        } else {
            plaintextQuery[counter] = char2Poly(queryString[i]);
            counter++;
        }
    }
    for (unsigned long i = counter; i < wordLength; i++) {
        plaintextQuery[i] = char2Poly(wcChar);
        plaintextE[i] = 1;
    }
    for (unsigned long i = wordLength; i < numSlots - numEmpty; i++) {
        plaintextQuery[i] = plaintextQuery[i % wordLength];
        plaintextE[i] = plaintextE[i % wordLength];
    }
#endif
    plaintextQuery.resize(numSlots, NTL::ZZX(0));
    plaintextE.resize(numSlots, NTL::ZZX(0));

    std::vector<NTL::ZZX> plaintextResult((unsigned long) numSlots - numEmpty, NTL::ZZX(1));
    plaintextResult.resize(numSlots, NTL::ZZX(0));

    // for (unsigned long i = 0; i < plaintextAttr.size(); i++) {
    //     // std::cout << poly2Char(plaintextAttr[i]) << ", ";
    // }
    // // std::cout << std::endl;
    // for (unsigned long i = 0; i < plaintextQuery.size(); i++) {
    //     // std::cout << poly2Char(plaintextQuery[i]) << ", ";
    // }
    // // std::cout << std::endl;
    // for (unsigned long i = 0; i < plaintextE.size(); i++) {
    //     // std::cout << plaintextE[i] << ", ";
    // }
    // // std::cout << std::endl;

    // for(unsigned long i = wordLength; i < 2*wordLength; i++) {
    //     // std::cout << poly2Char(plaintextAttr[i]) << ", ";
    // }
    // // std::cout << std::endl;
    // for(unsigned long i = wordLength; i < 2*wordLength; i++) {
    //     // std::cout << poly2Char(plaintextQuery[i]) << ", ";
    // }
    // // std::cout << std::endl;
    // for(unsigned long i = wordLength; i < 2*wordLength; i++) {
    //     // std::cout << plaintextE[i] << ", ";
    // }
    // // std::cout << std::endl;

    std::clog << wordLength << ", " << queryLength << ", " << numWords << ", ";

    // std::cout << "Plaintext Processing Done!" << std::endl;

    // Initialize and encrypt ciphertexts
    Ctxt ciphertextAttr(publicKey);
    Ctxt ciphertextQuery(publicKey);
    Ctxt tempCiphertext(publicKey);
    Ctxt ciphertextE(publicKey);
    Ctxt ciphertextResult(publicKey);
    Ctxt conjResult(publicKey);
//        Ctxt conjQuery(publicKey);
//        Ctxt conjCtxt(publicKey);

    ea.encrypt(ciphertextAttr, publicKey, plaintextAttr);
    ea.encrypt(ciphertextQuery, publicKey, plaintextQuery);
    ea.encrypt(ciphertextE, publicKey, plaintextE);
//        ea.encrypt(conjCtxt, publicKey, plaintextConjunction);
//        ea.encrypt(conjQuery, publicKey, plaintextConjunction);

    Ctxt oneCiphertext(publicKey);
    Ctxt zeroCiphertext(publicKey);

    std::vector<NTL::ZZX> onePlaintext(numSlots, NTL::ZZX(1));

    ea.encrypt(oneCiphertext, publicKey, onePlaintext);
    zeroCiphertext = oneCiphertext;
    zeroCiphertext -= oneCiphertext;

    // Initialize components
    std::vector<Ctxt> ciphertextWs, ciphertextRs, ciphertextSs, ciphertextDs;
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

    Ctxt test(publicKey), test1(publicKey);
    test = oneCiphertext;
    test1 = oneCiphertext;

    // std::cout << std::endl;
    startTime = std::chrono::high_resolution_clock::now();
    fastPower(test, plaintextDegree);
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    // std::cout << timeTaken.count() << ", ";

    startTime = std::chrono::high_resolution_clock::now();
    oneTotalProduct(test1, test, wordLength, ea);
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    // std::cout << timeTaken.count() << ", ";
    // std::cout << std::endl;

    // For compound conjunction queries
    std::vector<Ctxt> ciphertextConj;
    std::vector<Ctxt> ciphertextConjResult;
    for (unsigned long i = 0; i < conjSize; i++) {
        ciphertextConj.push_back(ciphertextAttr);
        ciphertextConjResult.push_back(ciphertextAttr);
    }

    // std::cout << "Encryption Done!" << std::endl;

    NTL::ZZX selectPoly;
    NTL::ZZX queryMask, finalMask;
    makeMask(selectPoly, 1, 1, wordLength, ea);
    makeMask(finalMask, wordLength, 1, wordLength, ea);

    // Step 1: Shift the attributes
    //     Remnants of experiment to pack differently, all characters of the same slot first
    //     But the shift time is much longer than packing word by word

    //     startTime = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    //     for(unsigned long i = 1; i < ciphertextWs.size(); i++) {
    //             ea.shift(ciphertextWs[i],-i*numWords);
    //     }
    //     endTime = std::chrono::high_resolution_clock::now();
    //     timeTaken = endTime-startTime;
    //     totalTime += timeTaken.count();
    //     // std::cout << "Pre-compute Time: " << timeTaken.count() << std::endl;

    startTime = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (unsigned long i = 1; i < ciphertextWs.size(); i++) {
#if new_simd
        ea.shift(ciphertextWs[i], (-i*numWords));
#else
        simdShift(ciphertextWs[i], ciphertextAttr, -i, wordLength, ea);
#endif   
    }
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    // totalTime += timeTaken.count();
    // std::cout << "Pre-compute Time: " << timeTaken.count() << std::endl;
    std::clog << timeTaken.count() << ", ";

    // for(unsigned long i = 0; i < ciphertextWs.size(); i++) {
    //         ea.decrypt(ciphertextWs[i],secretKey,plaintextResult);
    //         // std::cout << poly2Char(plaintextResult[0]) << ", ";
    // }
    // // std::cout << std::endl;

    // Step 2: Test if the characters are the same
    startTime = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (unsigned long i = 0; i < ciphertextWs.size() + conjSize; i++) {
        if (i < ciphertextWs.size()) {
            ciphertextWs[i] += ciphertextQuery;
        } else {
            ciphertextConj[i - ciphertextWs.size()] += ciphertextAttr;
        }
    }
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    totalTime += timeTaken.count();
    // std::cout << "XOR Time: " << timeTaken.count() << std::endl;

    startTime = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (unsigned long i = 0; i < ciphertextWs.size() + conjSize; i++) {
        if (i < ciphertextWs.size()) {
            equalTest(ciphertextRs[i], zeroCiphertext, ciphertextWs[i], plaintextDegree);
        } else {
            equalTest(ciphertextConjResult[i - ciphertextWs.size()], zeroCiphertext,
                      ciphertextConj[i - ciphertextWs.size()], plaintextDegree);
        }
        // }
        // for(unsigned long i = 0; i < ciphertextRs.size(); i++) {
        if (i < ciphertextWs.size()) {
            ciphertextRs[i] += ciphertextE;
        }
    }
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    totalTime += timeTaken.count();
    // std::cout << "Level left: " << ciphertextRs[1].capacity() << ", Equality Check + eMask Time: " << timeTaken.count() << std::endl;
    std::clog << timeTaken.count() << ", ";

    // Step 3: Combine results of character tests per shift
    startTime = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (unsigned long i = 0; i < ciphertextRs.size() + conjSize; i++) {
        if (i < ciphertextRs.size()) {
            oneTotalProduct(ciphertextSs[i], ciphertextRs[i], wordLength, ea);
        } else if (conjSize > 0) {
            oneTotalProduct(ciphertextConj[i - ciphertextRs.size()], ciphertextConjResult[i - ciphertextRs.size()],
                            wordLength, ea);
        }
    }
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    totalTime += timeTaken.count();
    // std::cout << "Level left: " << ciphertextSs[1].capacity() << ", Product of Equalities: " << timeTaken.count() << std::endl;
    std::clog << timeTaken.count() << ", ";

    // Step 4: Combine results from testing every possible shift
    startTime = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (unsigned long i = 0; i < ciphertextSs.size(); i++) {
        ciphertextSs[i].multiplyBy(ciphertextDs[i]);
        ciphertextSs[i].addConstant(selectPoly);
    }
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    totalTime += timeTaken.count();
    // std::cout << "Level left: " << ciphertextSs[1].capacity() << ", Disjunction Prep Time: " << timeTaken.count() << std::endl;
    std::clog << timeTaken.count() << ", ";

    startTime = std::chrono::high_resolution_clock::now();
    treeMult(ciphertextResult, ciphertextSs);
    ciphertextResult.addConstant(selectPoly);

    if (conjSize > 0) {
        treeMult(conjResult, ciphertextConj);
        ciphertextResult.multiplyBy(conjResult);
    }
    // ciphertextResult.multByConstant(finalMask);
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    totalTime += timeTaken.count();
    // std::cout << "Level left: " << ciphertextResult.capacity() << ", Disjunction Time: " << timeTaken.count() << std::endl;
    std::clog << timeTaken.count() << ", ";

    std::clog << totalTime << ", ";

    startTime = std::chrono::high_resolution_clock::now();
    tempCiphertext = ciphertextResult;
#pragma omp parallel for
    for (unsigned long i = 1; i < wordLength; i++) {
        ciphertextWs[i] = ciphertextResult;
        ea.shift(ciphertextWs[i], i);
    }
    for (unsigned long i = 1; i < wordLength; i++) {
        tempCiphertext += ciphertextWs[i];
    }
    tempCiphertext.multiplyBy(ciphertextAttr);
    endTime = std::chrono::high_resolution_clock::now();
    timeTaken = endTime - startTime;
    // totalTime += timeTaken.count();
    // std::cout << "Level left: " << tempCiphertext.capacity() << ", Fill + Selection Time: " << timeTaken.count() << std::endl;
    std::clog << timeTaken.count() << ", ";

    // std::cout << "Total Time: " << totalTime << std::endl;

    ea.decrypt(ciphertextResult, secretKey, plaintextResult);
    for (unsigned long i = 0; i < wordLength; i++) {
        // std::cout << poly2Long(plaintextResult[i]) << ", ";
    }
    // std::cout << std::endl;

    ea.decrypt(tempCiphertext, secretKey, plaintextResult);
    for (unsigned long i = 0; i < wordLength; i++) {
        // std::cout << poly2Char(plaintextResult[i]) << ", ";
    }
    // std::cout << std::endl;

    std::clog << poly2Char(plaintextResult[0]) << ", ";

//    ea.decrypt(ciphertextSs[0], secretKey, plaintextResult);
//    for (unsigned long i = 0; i < wordLength; i++) {
//        // std::cout << poly2Long(plaintextResult[i]) << ", ";
//    }
//    // std::cout << std::endl;
//
//    ea.decrypt(ciphertextRs[0], secretKey, plaintextResult);
//    for (unsigned long i = 0; i < wordLength; i++) {
//        // std::cout << poly2Long(plaintextResult[i]) << ", ";
//    }
//    // std::cout << std::endl;
//
//    ea.decrypt(ciphertextWs[0], secretKey, plaintextResult);
//    for (unsigned long i = 0; i < wordLength; i++) {
//        // std::cout << poly2Long(plaintextResult[i]) << ", ";
//    }
//    // std::cout << std::endl;

//        ea.decrypt(conjResult, secretKey, plaintextResult);
//        for(unsigned long i = 0; i < wordLength; i++) {
//            // std::cout << poly2Long(plaintextResult[i]) << ", ";
//        }
//        // std::cout << std::endl;

//        for(unsigned long i = 0; i < conjSize; i++) {
//            ea.decrypt(ciphertextConj[i],secretKey, plaintextResult);
//            for(unsigned long i = 0; i < wordLength; i++) {
//                // std::cout << poly2Long(plaintextResult[i]) << ", ";
//            }
//            // std::cout << std::endl;
//        }
//        // std::cout << std::endl;
    std::clog << std::endl;
}
