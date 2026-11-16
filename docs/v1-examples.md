# V1 examples

End-to-end run of the V1 pipeline (vector top-5 retrieval, `qwen3-embedding:0.6b` for retrieval, `qwen3.5:9b` for generation) against the approved baseline `English-Vocabulary-in-Use-Pre-Intermediate.pdf` (265 pages, 512-token chunks, overlap 64).

- doc_id: `3a357be041df101c`
- chunks indexed: 203

## Q1: What is the difference between 'borrow' and 'lend'?

**Answer:** According to the text, the difference between **lend** and **give** is that if you lend something to someone, they have to give it back, whereas if you give something to them, they can keep it. The text does not explicitly define the difference between *borrow* and *lend*, but it provides examples of how to use them in requests:

*   "Could I **borrow** your dictionary?" [1]
*   "Could you **lend** me a dictionary?" [2]
*   "If you **lend** something to someone, they have to give it back. If you **give** something to them, they can keep it." [1]

Based on the context of these sentences and the definition of *borrow* in exercise 31.1 ("Match the words... borrow... a dictionary"), borrowing implies taking something with the intention of returning it, while lending implies giving something with the expectation that it will be returned.

**Abstained:** False

**Citations:**

- `3a357be041df101c:47` — pages 69–71: “things in the classroom. I'd like you to work with a partner [someone else, i. e. another student]. Henrique, could you swap placeswith Lorena? [change seats / …”
- `3a357be041df101c:168` — pages 221–222: “something out 5 because you haven't got one (or someone else hasn't got one) 6 to highlight something 7 to sharpen a pencil 8 to measure something, or perhaps t…”

**Retrieval:** configuration `v1`, candidates `3a357be041df101c:47`, `3a357be041df101c:168`, `3a357be041df101c:101`, `3a357be041df101c:30`, `3a357be041df101c:100`, scores [0.5595, 0.5155, 0.4525, 0.4346, 0.4285]

## Q2: What is the difference between 'job' and 'work'?

**Answer:** Tài liệu không có đủ thông tin để trả lời câu hỏi này vì các đoạn văn bản chỉ định nghĩa và sử dụng từ "job" (công việc) nhưng không đưa ra sự phân biệt cụ thể giữa "job" và "work".

**Abstained:** False

**Citations:**


**Retrieval:** configuration `v1`, candidates `3a357be041df101c:54`, `3a357be041df101c:58`, `3a357be041df101c:57`, `3a357be041df101c:59`, `3a357be041df101c:55`, scores [0.5445, 0.5081, 0.491, 0.4608, 0.4386]

## Q3: What does the book say a phrasal verb is?

**Answer:** According to the text, a phrasal verb is defined as "a verb + adverb or preposition, and sometimes a verb + adverb and preposition" [3].

**Abstained:** False

**Citations:**

- `3a357be041df101c:120` — pages 164–165: “correct word to complete the sentences. 1 I'm afraid I deleted your email on / in / by accident. 2 Did she hit him on / by / in purpose? 3 I just saw them with …”

**Retrieval:** configuration `v1`, candidates `3a357be041df101c:123`, `3a357be041df101c:1`, `3a357be041df101c:120`, `3a357be041df101c:122`, `3a357be041df101c:128`, scores [0.6509, 0.6313, 0.5739, 0.5568, 0.5359]
