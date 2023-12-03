resource "aws_s3_bucket" "awsdocs" {
  bucket = var.name_s3
  acl    = "private"


  tags = {
    Name = var.name_s3
  }
}
