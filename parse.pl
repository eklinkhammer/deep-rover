#!/usr/bin/perl

my $counter = 0;
while (<>) {
    print $_;
    $counter = $counter + 1;
    if ($counter > 1) {
	break;
    }
}
