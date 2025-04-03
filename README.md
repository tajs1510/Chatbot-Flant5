# Chatbot-Flant5

1.	Khái niệm mô hình Flan-T5
Flan-T5 là phiên bản cải tiến của mô hình T5 (Text-to-Text Transfer Transformer), được phát triển bởi Google. Mô hình này đã được tinh chỉnh trên một tập hợp đa dạng các nhiệm vụ, giúp cải thiện khả năng tổng quát và hiệu suất trên nhiều tác vụ xử lý ngôn ngữ tự nhiên mà không cần huấn luyện lại cho từng nhiệm vụ riêng biệt. [1]

 
2.	Nguyên lý của Flan-T5 hoạt động
FLAN-T5 sử dụng kiến trúc Transformer, một mô hình học sâu được Vaswani và cộng sự giới thiệu vào năm 2017, có khả năng xử lý hiệu quả các chuỗi dữ liệu như văn bản nhờ cơ chế tự chú ý, cho phép mô hình hiểu được mối quan hệ giữa các phần tử bất kể vị trí của chúng. Dưới đây là các bước chi tiết về một mô hình Flan-T5 hoạt động:

Chuẩn bị cho dịch văn bản: pip install transformers datasets evaluate sacrebleu.
Chuyển đổi các bài toán dịch thành định dạng text-to-text, tức là tạo ra những prompt cụ thể cho việc dịch (ví dụ: "Dịch đoạn văn sau sang tiếng Anh: ...").

Tải bộ dữ liệu OPUS Books (dùng cho dịch thuật):Giúp mô hình có nhiều ví dụ dịch thuật thực tế và đa dạng.
from datasets import load_dataset

books = load_dataset("opus_books", "en-fr")
Tiền xử lý dữ liệu: 

•	Thu thập dữ liệu từ đa tác vụ: Sưu tập các bộ dữ liệu từ các nguồn khác nhau, bao gồm tác vụ dịch thuật, tóm tắt, hỏi đáp, phân loại, v.v. Những tác vụ càng đa dạng giúp mô hình tổng quát hóa tốt hơn.

•	Định dạng dưới dạng text to text: Mỗi tác vụ được chuyển hóa thành định dạng đầu vào là một chỉ dẫn (prompt) và đầu ra mong đợi là văn bản.

Xử lý khi huấn luyện và suy luận: rong quá trình huấn luyện, FLAN-T5 được đào tạo theo phương pháp teacher forcing. 

Quá trình huấn luyện mô hình: 

•	Giai đoạn 1 – Tiền huấn luyện (pre-training) T5 gốc.

•	Giai đoạn 2 – Fine-tuning theo hướng dẫn (instruction tuning).

•	Kỹ thuật huấn luyện cải tiến. Để đạt hiệu quả cao nhất, nhóm FLAN còn áp dụng một số kỹ thuật dữ liệu và huấn luyện nâng cao. 1. Kết hợp cả dạng hướng dẫn zero-shot (chỉ đưa ra yêu cầu) và few-shot (đưa thêm vài ví dụ mẫu trong prompt). 2. Sử dụng một phần chain-of-thought prompting. 3. Input inversion. 4. Balancing task mixtures 
Kết quả và mô hình thu được: Sau khi hoàn thành huấn luyện với tập “FLAN Collection” đồ sộ, Google công bố các trọng số mô hình FLAN-T5 cho 5 kích cỡ: Small (~80 triệu tham số), Base (~250M), Large (~800M), XL (~3 tỷ) và XXL (~11 tỷ tham số) 
Quá trình huấn luyện FLAN-T5 gồm hai bước: (1) Tiền huấn luyện trên dữ liệu tổng quát để có nền tảng ngôn ngữ; (2) Fine-tune trên tập hợp hàng ngàn nhiệm vụ dưới dạng hướng dẫn để mô hình trở nên linh hoạt, hiểu ý định người dùng và giải nhiều bài toán khác nhau. Chính bước thứ hai độc đáo này đã làm nên sức mạnh của FLAN-T5 so với mô hình T5 truyền thống.

Dữ liệu đầu vào (Input) là dạng câu hỏi( hoặc yêu cầu) dưới dạng văn bản tự nhiên. Ví dụ:

•	"What is Natural Language Processing (NLP)?"

•	"Explain machine learning with examples."

•	"How does deep learning work?"

Đây là dữ liệu ngôn ngữ tự nhiên (Natural Language Data) do người dùng gửi lên từ frontend(React)

Dữ liều đầu ra (Output) là câu trả lời dạng văn bản hoàn chỉnh, chi tiết, dễ hiểu. Ví dụ:

•	"Natural Language Processing (NLP) is a branch of AI that focuses on the interaction between computers and human language..."

•	Một đoạn văn dài , giải thích, có thể kèm ví dụ, diễn giải tùy theo múc độ câu hỏi. 

Dữ liệu liệu huấn luyên của mô hình Flan-T5, Flan-T5 là mô hình “pre-trained”, đã được Google huấn luyện trên hàng loạt tập dữ liệu lớn, như:

Tên bộ dữ liệu	Dùng để làm gì	Đặc điểm nổi bật

C4 (Colossal Clean Crawled Corpus)	Dữ liệu web tổng hợp để hiểu ngôn ngữ tổng quát	Hàng trăm GB dữ liệu văn bản từ internet, sạch và chuẩn

Eli5 (Explain Like I’m 5)	Trả lời câu hỏi phức tạp đơn giản hóa	Giúp mô hình trả lời đơn giản dễ hiểu, giống như giải thích cho trẻ em 5 tuổi

TriviaQA, Natural Questions	Trả lời các câu hỏi kiến thức thật	Dữ liệu hỏi đáp từ Google Search, Wikipedia,...

SQuAD	Câu hỏi đọc hiểu từ đoạn văn bản	Học cách tìm thông tin từ ngữ cảnh cụ thể

MultiNLI, BoolQ, RTE	Hiểu ngữ cảnh, suy luận logic	Hiểu câu chuyện, mối liên hệ logic giữa các câu

![image](https://github.com/user-attachments/assets/77154ecf-f46e-49be-99ee-caea79697bce)

