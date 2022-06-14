import pkg_resources
import os
from re import search

offset = '    '

def control_dep_string(deps):
    return "with tf.control_dependencies([{deps}]):".format(deps=deps)


def time_stamp_string(stamp_id):
    return "stime{id} = tf.timestamp()".format(id=stamp_id)


def matmul_string_gemm(inp, out):
    return "{out} = tf.matmul({inp1}, {inp2})".format(out=out, inp1=inp[0], inp2=inp[1])


def matmul_string_gemv(inp, out):
    return "{out} = tf.matmul(tf.transpose({inp1}), {inp2})".format(out=out, inp1=inp[0], inp2=inp[1])


def matmul_string_trsm(inp, out):
    return "{out} = tf.matmul(tf.linalg.inv({inp1}), {inp2})".format(out=out, inp1=inp[0], inp2=inp[1])


def matmul_string_trsv(inp, out):
    return "{out} = tf.matmul(tf.linalg.inv({inp1}), {inp2})".format(out=out, inp1=inp[0], inp2=inp[1])


def matmul_string_syrk(inp, out):
    return "{out} = tf.matmul(tf.transpose({inp1}), {inp2})".format(out=out, inp1=inp, inp2=inp)


def cholesky_string(inp, out):
    return "{out} = tf.linalg.cholesky({inp})".format(out=out, inp=inp)


def event_log_activity_labels(stamp_id, activity_name):
    code = ""
    code += offset
    # code += "event{} = [id, "{}", timestamps[{}].numpy(), timestamps[{}].numpy(), dims, num_threads]".format(stamp_id, activity_name, stamp_id, stamp_id+1)
    code += "event{} = [id, ".format(stamp_id)
    code += '"'
    code += activity_name
    code += '"'
    code += ", timestamps[{}].numpy(), timestamps[{}].numpy(), dims, num_threads]".format(stamp_id, stamp_id+1)
    code += "\n"
    code += offset + "csv_writer.writerow(event{})".format(stamp_id)
    code += "\n"
    return  code


def generate_tf_matmul_gemm(inp,out,stamp_id):
    code = ""
    code += offset+"{}\n".format(control_dep_string("stime{}".format(stamp_id)))
    code += offset+offset+"{}\n".format(matmul_string_gemm(inp,out))
    code += offset+"{}\n".format(control_dep_string(out))
    code += offset+offset+"{}\n".format(time_stamp_string(stamp_id+1))
    return  code

def generate_tf_matmul_gemv(inp,out,stamp_id):
    code = ""
    code += offset+"{}\n".format(control_dep_string("stime{}".format(stamp_id)))
    code += offset+offset+"{}\n".format(matmul_string_gemv(inp,out))
    code += offset+"{}\n".format(control_dep_string(out))
    code += offset+offset+"{}\n".format(time_stamp_string(stamp_id+1))
    return  code

def generate_tf_matmul_trsm(inp,out,stamp_id):
    code = ""
    code += offset+"{}\n".format(control_dep_string("stime{}".format(stamp_id)))
    code += offset+offset+"{}\n".format(matmul_string_trsm(inp,out))
    code += offset+"{}\n".format(control_dep_string(out))
    code += offset+offset+"{}\n".format(time_stamp_string(stamp_id+1))
    return  code

def generate_tf_matmul_trsv(inp,out,stamp_id):
    code = ""
    code += offset+"{}\n".format(control_dep_string("stime{}".format(stamp_id)))
    code += offset+offset+"{}\n".format(matmul_string_trsv(inp,out))
    code += offset+"{}\n".format(control_dep_string(out))
    code += offset+offset+"{}\n".format(time_stamp_string(stamp_id+1))
    return  code

def generate_tf_matmul_syrk(inp,out,stamp_id):
    code = ""
    code += offset+"{}\n".format(control_dep_string("stime{}".format(stamp_id)))
    code += offset+offset+"{}\n".format(matmul_string_syrk(inp,out))
    code += offset+"{}\n".format(control_dep_string(out))
    code += offset+offset+"{}\n".format(time_stamp_string(stamp_id+1))
    return  code

def generate_tf_linalg_cholesky(inp,out,stamp_id):
    code = ""
    code += offset+"{}\n".format(control_dep_string("stime{}".format(stamp_id)))
    code += offset+offset+"{}\n".format(cholesky_string(inp,out))
    code += offset+"{}\n".format(control_dep_string(out))
    code += offset+offset+"{}\n".format(time_stamp_string(stamp_id+1))
    return  code




if __name__ == '__main__':

    for i in range(5):
        tf_path = "tf-code.py"
        julia_path = "algorithm{variant_number}.jl".format(variant_number = i)

        f = open(julia_path, 'r')
        lines = f.readlines()

        id = 0
        code = ""
        # I added this part of the code.
        result = ""
        code1 = ""

        for line in lines:

            if search("return", line):
                temp = line.split("(")[1].split(")")[0]
                print("temp value is: " + temp)
                result += temp

            elif search("gemm!", line):
                inp = line.split("!")[1].split(",")[3:5]
                inp = [x.strip() for x in inp]     #removes spaces
                print(inp)

                out = line.split("!")[1].split(",")[-1].split(")")[0].strip()
                print(out)

                code += generate_tf_matmul_gemm(inp,out,id)
                code1 += event_log_activity_labels(id, "gemm")
                id =id+1
                code += "\n"
                code1 += "\n"

            elif search("gemv!", line):
                inp = line.split("!")[1].split(",")[2:4]
                inp = [x.strip() for x in inp]     #removes spaces
                print(inp)

                out = line.split("!")[1].split(",")[-1].split(")")[0].strip()
                print(out)

                code += generate_tf_matmul_gemv(inp,out,id)
                code1 += event_log_activity_labels(id, "gemv")
                id = id + 1
                code += "\n"
                code1 += "\n"

            elif search("trsm!", line):
                inp = line.split("!")[1].split(",")[5:7]
                inp = [x.strip() for x in inp]     #removes spaces
                inp[1] = inp[1].split(")")[0]
                print(inp)

                out = line.split("!")[1].split(",")[-1].split(")")[0].strip()
                print(out)

                code += generate_tf_matmul_trsm(inp,out,id)
                code1 += event_log_activity_labels(id, "trsm")
                id = id + 1
                code += "\n"
                code1 += "\n"

            elif search("trsv!", line):
                inp = line.split("!")[1].split(",")[3:5]
                inp = [x.strip() for x in inp]     #removes spaces
                inp[1] = inp[1].split(")")[0]
                print(inp)

                out = line.split("!")[1].split(",")[-1].split(")")[0].strip()
                print(out)

                code += generate_tf_matmul_trsv(inp,out,id)
                code1 += event_log_activity_labels(id, "trsv")
                id = id + 1
                code += "\n"
                code1 += "\n"

            elif search("syrk!", line):
                inp = line.split("!")[1].split(",")[3].strip()   #removes spaces
                print(inp)

                out = line.split("!")[1].split(",")[-1].split(")")[0].strip()
                print(out)

                code += generate_tf_matmul_syrk(inp,out,id)
                code1 += event_log_activity_labels(id, "syrk")
                id = id + 1
                code += "\n"
                code1 += "\n"

            elif search("LAPACK.potrf!", line):
                #inp = line.split("!")[1].split(",")[1:2]
                #inp = [x.strip() for x in inp]
                #inp[0] = inp[0].split(")")[0]
                inp = line.split("!")[1].split(",")[-1].split(")")[0].strip()
                print(inp)

                out = line.split("!")[1].split(",")[-1].split(")")[0].strip()
                print(out)

                code += generate_tf_linalg_cholesky(inp,out,id)
                code1 += event_log_activity_labels(id, "LAPACK.potrf")
                id = id + 1
                code += "\n"
                code1 += "\n"

        f.close()

        time_stamp_str = ""
        time_stamp_str += "["
        for j in range(id+1):
            time_stamp_str += "stime{},".format(j)
        time_stamp_str = time_stamp_str[:-1]
        time_stamp_str += "]"

        #print(code)
        #print(time_stamp_str)

        inject = {
            "code":code,
            "timestamps":time_stamp_str,
            "result":result,
            "code1":code1
        }

        template_str = pkg_resources.resource_string(__name__, tf_path).decode("UTF-8")
        #print(template_str.format(**inject))

        outfile = "output_tf_{variant_number}.py".format(variant_number=i)
        with open(outfile, "wt", encoding='utf-8') as output_file:
            print("Writing", outfile)
            output_file.write(template_str.format(**inject))

