# Git Rules
1. Fetch và Pull trước khi bắt đầu làm
2. Chỉ Push sau khi Pull
3. Không merge vào main/dev
4. dev là branch chính để phát triển, main chỉ dùng để release
5. Mỗi khi cần thực hiện phát triển 1 tính năng hay một thay đổi lớn, hãy tạo một branch mới từ origin/dev để làm việc trên đó. Quy tắc đặt tên nhánh:
    - Tính năng: `feature/ten-tinh-nang`
    - Sửa lỗi: `bugfix/ten-loi`
    - Cải tiến: `improvement/ten-cai-tien`
6. Khi hoàn thành công việc trên branch, hãy tạo một pull request (PR) để merge vào dev. PR phải được review và approved bởi ít nhất 1 thành viên khác trước khi merge.
7. Luôn giữ cho commit messages rõ ràng và có ý nghĩa, mô tả ngắn gọn về những thay đổi đã thực hiện. Quy tắc:
    - Commit những thay đổi nhỏ và có liên quan, tránh commit quá nhiều thay đổi trong một lần.
    - Tên commit: `type(scope): mô tả ngắn gọn` + description chi tiết (nếu cần) (English)
    - Type: `feat` (tính năng), `fix` (sửa lỗi), `docs` (tài liệu), `style` (định dạng), `refactor` (cải tiến), `test` (thêm/sửa test), `chore` (công việc khác)
    - Scope: phần của dự án bị ảnh hưởng (ví dụ: observation, agent, utils)
    - Ví dụ: `feat(observation): thêm chức năng mới cho observation` hoặc `fix(agent): sửa lỗi trong logic agent`
8. Trước khi merge vào dev, hãy đảm bảo rằng tất cả các test đã được chạy và pass thành công, folder test: `tests/`.
9. WORKLOG.md và JOURNAL.md chỉ được cập nhật khi ở trong branch `dev`. Sau đó tạo PR tới `main`.

